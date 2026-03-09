/*  This file is part of the OpenLB library
 *
 *  Copyright (C) 2022 Louis Kronberg, Pavel Eichler, Stephan Simonis
 *  E-mail contact: info@openlb.net
 *  The most recent release of OpenLB can be downloaded at
 *  <http://www.openlb.net/>
 *
 *  This program is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU General Public License
 *  as published by the Free Software Foundation; either version 2
 *  of the License, or (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public
 *  License along with this program; if not, write to the Free
 *  Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
 *  Boston, MA  02110-1301, USA.
*/

#ifndef DYNAMICS_COLLISION_CUM_H
#define DYNAMICS_COLLISION_CUM_H

#include "descriptor/definition/cum.h"

namespace olb {

template <typename DESCRIPTOR>
struct cum {
  static_assert(std::is_same<typename DESCRIPTOR::category_tag,
                             descriptors::tag::CUM>::value,
                "DESCRIPTOR is tagged as CUM");

  /// [STAGE 1 實作] Forward Central Moment Transformation (Chimera 前向變換)
  /// 將 well-conditioned populations f̄_ijk = f_ijk - w_ijk 轉換為 central moments κ_abc
  ///
  /// 運算方向：z → y → x（從最後一個維度開始）
  /// 每個方向做 9 組三元組 (a,b,c) 線性變換，
  /// 減去流體速度 u 的影響，得到以流體為參考系的 moments。
  ///
  /// 迴圈依 i 的值自動對應不同方向的方程式：
  ///   i=2 (z方向, w)：Eq. J.4, J.5, J.6
  ///   i=1 (y方向, v)：Eq. J.7, J.8, J.9
  ///   i=0 (x方向, u)：Eq. J.10, J.11, J.12
  ///
  /// 每組三元組的公式統一為（以 x 方向為例）：
  ///   momenta[a] = (f₁ + f₋₁) + f₀            → κ_{0βγ}   (J.4/J.7/J.10) 
  ///   momenta[b] = (f₁ - f₋₁) - u·(κ₀ + k)   → κ_{1βγ}   (J.5/J.8/J.11)
  ///   momenta[c] = (f₁ + f₋₁) - 2u·(f₁ - f₋₁) + u²·(κ₀ + k)
  ///                                           → κ_{2βγ}   (J.6/J.9/J.12)
  ///
  /// 其中 k = K_{αβγ}·(1-δρ)，K 由 weights 計算：
  ///   K_{ij|γ} = Σ_k k^γ · w_{ijk}   (J.13)
  ///   K_{i|βγ} = Σ_j j^β · K_{ij|γ}  (J.14)
  ///   K_{αβγ}  = Σ_i i^α · K_{i|βγ}  (J.15)
  /// K 陣列定義於 cum.h 的 K<3,27>。
  template <typename MOMENTA, typename CELL, typename U,
            typename V = typename CELL::value_t>
  static void computeMomenta(MOMENTA& momenta, CELL& cell, U& u) any_platform
  {
    // initialize momenta with the populations
    for (int i = 0; i < DESCRIPTOR::q; i++) {
      momenta[i] = cell[i];
    }
    constexpr auto passes = DESCRIPTOR::q / DESCRIPTOR::d;
    for (int i = DESCRIPTOR::d - 1; i >= 0; i--) {
      // i=2: z方向 → Eq. J.4–J.6
      // i=1: y方向 → Eq. J.7–J.9
      // i=0: x方向 → Eq. J.10–J.12
      for (int j = 0; j < passes; j++) {
        auto [a, b, c] = descriptors::cum_data::velocityIndices< DESCRIPTOR::d, DESCRIPTOR::q>[passes * i + j];

        V k = descriptors::constantK<V, DESCRIPTOR::d, DESCRIPTOR::q>(passes * i + j); // k = K_{αβγ}·(1-δρ)  (J.13–J.15)
        V sum        = momenta[a] + momenta[c];   // f₁ + f₋₁
        V difference = momenta[c] - momenta[a];   // f₁ - f₋₁
        //0階變換(x,y,z變數逐次此用)
        momenta[a]   = momenta[a] + momenta[b] + momenta[c];  // (J.4/J.7/J.10) κ₀ = (f₁+f₋₁) + f₀
        //1階變換
        momenta[b]   = difference - (momenta[a] + k) * u[i];  // (J.5/J.8/J.11) κ₁ = (f₁-f₋₁) - u·(κ₀+k)
        //2階變換
        momenta[c]   =sum - V(2) * difference * u[i] + u[i] * u[i] * (momenta[a] + k); // (J.6/J.9/J.12) κ₂ = (f₁+f₋₁) - 2u·(f₁-f₋₁) + u²·(κ₀+k)
      }
    }
  }

  /// [STAGE 5 實作] Backward Central Moment Transformation (Chimera 反向變換)
  /// 將 post-collision central moments κ*_abc 轉回 well-conditioned populations f̄*_ijk
  ///
  /// 運算方向：x → y → z（與 Stage 1 相反）
  /// 公式是 Stage 1 的逆運算。
  ///
  /// 迴圈依 i 的值自動對應不同方向的方程式：
  ///   i=0 (x方向, u)：Eq. J.20, J.21, J.22
  ///   i=1 (y方向, v)：Eq. J.23, J.24, J.25
  ///   i=2 (z方向, w)：Eq. J.26, J.27, J.28
  ///
  /// 每組三元組的公式統一為（以 x 方向為例）：
  ///   ma = f̄*₋₁ = ((κ₀+k)·(u²-u) + κ₁·(2u-1) + κ₂) / 2   (J.21/J.24/J.27)
  ///   mb = f̄*₀  = κ₀·(1-u²) - 2u·κ₁ - κ₂                  (J.20/J.23/J.26)
  ///   mc = f̄*₁  = ((κ₀+k)·(u²+u) + κ₁·(2u+1) + κ₂) / 2   (J.22/J.25/J.28)
  template <typename MOMENTA, typename CELL, typename U,
            typename V = typename CELL::value_t>
  static void computePopulations(MOMENTA& momenta, CELL& cell,
                                 U& u) any_platform
  {

    constexpr auto passes = DESCRIPTOR::q / DESCRIPTOR::d;
    for (int i = 0; i < DESCRIPTOR::d; i++) {
      // i=0: x方向 → Eq. J.20–J.22
      // i=1: y方向 → Eq. J.23–J.25
      // i=2: z方向 → Eq. J.26–J.28
      for (int j = 0; j < passes; j++) {
        auto [a, b, c] = descriptors::cum_data::velocityIndices<
            DESCRIPTOR::d, DESCRIPTOR::q>[passes * i + j];
        V k = descriptors::constantK<V, DESCRIPTOR::d, DESCRIPTOR::q>(
            passes * i + j);                // k = K_{αβγ}·(1-δρ)  (J.13–J.15)

        // (J.21/J.24/J.27) f̄*₋₁ = ((κ₀+k)·(u²-u) + κ₁·(2u-1) + κ₂) / 2
        V ma = ((momenta[c] - momenta[b]) * V(0.5) + momenta[b] * u[i] +
                (momenta[a] + k) * (u[i] * u[i] - u[i]) * V(0.5));
        // (J.20/J.23/J.26) f̄*₀ = κ₀·(1-u²) - 2u·κ₁ - κ₂
        V mb = (momenta[a] - momenta[c]) - V(2) * momenta[b] * u[i] -
               (momenta[a] + k) * u[i] * u[i];
        // (J.22/J.25/J.28) f̄*₁ = ((κ₀+k)·(u²+u) + κ₁·(2u+1) + κ₂) / 2
        V mc = ((momenta[c] + momenta[b]) * V(0.5) + momenta[b] * u[i] +
                (momenta[a] + k) * (u[i] * u[i] + u[i]) * V(0.5));

        momenta[a] = ma;
        momenta[b] = mb;
        momenta[c] = mc;
      }
    }

    // write back to cell
    for (int i = 0; i < DESCRIPTOR::q; i++) {
      cell[i] = momenta[i];
    }
  }

  //從這裡開始...................................................................


  template <typename CELL, typename U, typename V = typename CELL::value_t>
  static V cumCollision(CELL& cell, const V& omega, V& rho, U& u) any_platform
  {
    V drho        = rho - 1;      // δρ = ρ - 1  (J.2 的反算：論文寫 ρ = δρ+1)
    V inverse_rho = 1. / rho;      // 1/ρ
    V uSqr        = util::normSqr<V, DESCRIPTOR::d>(u);  // u²+v²+w²
    // 注意：δρ (J.1) 和 u,v,w (J.3) 由 OpenLB 框架的 computeRhoU() 在外部計算

    // ============================================================
    // STAGE 1: Forward Central Moment Transformation
    //          f̄_ijk (well-conditioned populations) → κ_αβγ (central moments)
    //          使用 Chimera 變換，沿 z→y→x 依序將 populations
    //          轉換為以流體速度 u 為參考系的 central moments。
    //          本實作為 well-conditioned 版本（cell 已減去 lattice weights）。
    //          (論文 Appendix J, Eq. J.4–J.12；
    //           原始版本見 Section 4.1, Eq. 43–45)
    // ============================================================
    V moments[DESCRIPTOR::q];
    computeMomenta(moments, cell, u);
    auto [mbbb, mabb, mbab, mbba, maab, macb, maba, mabc, mbaa, mbac, maaa,
          maac, maca, macc, mcbb, mbcb, mbbc, mccb, mcab, mcbc, mcba, mbcc,
          mbca, mccc, mcca, mcac, mcaa] = moments;

    // 鬆弛率定義（論文 Section 4.3, Eq. 55–80）
    // 程式碼變數名 → 論文符號對照：
    //   omega   = ω₁  (物理黏性，唯一影響 Navier-Stokes 的參數)
    //   omega2  = ω₂  (bulk viscosity, Eq. 63)
    //   omega3  = ω₃  (3rd order symmetric, Eq. 64–66)
    //   omega4  = ω₄  (3rd order antisymmetric, Eq. 67–69)
    //            此處也用於 ω₅ (Eq. 70, C₁₁₁)
    //   omega5  = ω₅  (未使用，由 omega4 代替)
    //   omega6  = ω₆  (4th order traceless, Eq. 71–72)
    //            此處也用於 ω₇ (Eq. 73) 和 ω₈ (Eq. 74–76)
    //   omega7  = ω₉  (5th order, Eq. 77–79)
    //   omega10 = ω₁₀ (6th order, Eq. 80)
    // 目前全部設為 1（完全鬆弛至平衡態），論文建議可在 {0..2} 範圍調整。
    const V omega2 =
        1; // ω₂: bulk viscosity (Eq. 63). If modified, constants A and B must be modified too!!!
    const V omega3  = 1;  // ω₃: 3rd order symmetric combinations (Eq. 64–66)
    const V omega4  = 1;  // ω₄: 3rd order antisymmetric + C₁₁₁ (Eq. 67–70)
    const V omega5  = 1;  // ω₅: (unused, omega4 used instead for Eq. 70)
    const V omega6  = 1;  // ω₆/ω₇/ω₈: 4th order (Eq. 71–76)
    const V omega7  = 1;  // ω₉: 5th order (Eq. 77–79)
    const V omega10 = 1;  // ω₁₀: 6th order (Eq. 80)

    // ============================================================
    // STAGE 2: Forward Cumulants Transformation
    //          κ_αβγ (central moments) → C_αβγ (cumulants)
    //          從 central moments 扣除低階 cumulant 的乘積項，
    //          得到真正的 cumulants。
    //          注意：1~3 階的 cumulant = central moment，不需轉換；
    //          4~6 階才需要扣除低階乘積。
    //          本實作為 well-conditioned 版本（含 +1/3, δρ/9ρ 修正項）。
    //          (論文 Appendix J, Eq. J.16–J.19；
    //           原始版本見 Section 4.2, Eq. 46–54)
    // ============================================================
    // --- 4th order cumulants ---
    // (Eq. J.16) C₂₁₁ = κ₂₁₁ - ((κ₂₀₀ + 1/3)·κ₀₁₁ + 2·κ₁₁₀·κ₁₀₁)/ρ
    V CUMcbb = mcbb - ((mcaa + 1. / 3) * mabb + 2 * mbba * mbab) * inverse_rho;
    // (Eq. J.16 permuted) C₁₂₁
    V CUMbcb = mbcb - ((maca + 1. / 3) * mbab + 2 * mbba * mabb) * inverse_rho;
    // (Eq. J.16 permuted) C₁₁₂
    V CUMbbc = mbbc - ((maac + 1. / 3) * mbba + 2 * mbab * mabb) * inverse_rho;

    // (Eq. J.17) C₂₂₀ = κ₂₂₀ - ((κ₂₀₀·κ₀₂₀ + 2·κ₁₁₀²) + (κ₂₀₀+κ₀₂₀)/3)/ρ + δρ/(9ρ)
    V CUMcca =
        mcca - (((mcaa * maca + 2 * mbba * mbba) + 1. / 3 * (mcaa + maca)) *
                    inverse_rho -
                1. / 9 * (drho * inverse_rho));
    // (Eq. J.17 permuted) C₂₀₂
    V CUMcac =
        mcac - (((mcaa * maac + 2 * mbab * mbab) + 1. / 3 * (mcaa + maac)) *
                    inverse_rho -
                1. / 9 * (drho * inverse_rho));
    // (Eq. J.17 permuted) C₀₂₂
    V CUMacc =
        macc - (((maac * maca + 2 * mabb * mabb) + 1. / 3 * (maac + maca)) *
                    inverse_rho -
                1. / 9 * (drho * inverse_rho));

    // --- 5th order cumulants ---
    // (Eq. J.18) C₁₂₂ = κ₁₂₂ - ((κ₀₀₂·κ₁₂₀ + κ₀₂₀·κ₁₀₂ + 4·κ₀₁₁·κ₁₁₁
    //            + 2·(κ₁₀₁·κ₀₂₁ + κ₁₁₀·κ₀₁₂)) + (κ₁₂₀+κ₁₀₂)/3) / ρ
    V CUMbcc = mbcc - ((maac * mbca + maca * mbac + 4 * mabb * mbbb +
                        2 * (mbab * macb + mbba * mabc)) +
                       1. / 3 * (mbca + mbac)) *
                          inverse_rho;
    // (Eq. J.18 permuted) C₂₁₂
    V CUMcbc = mcbc - ((maac * mcba + mcaa * mabc + 4 * mbab * mbbb +
                        2 * (mabb * mcab + mbba * mbac)) +
                       1. / 3 * (mcba + mabc)) *
                          inverse_rho;
    // (Eq. J.18 permuted) C₂₂₁
    V CUMccb = mccb - ((mcaa * macb + maca * mcab + 4 * mbba * mbbb +
                        2 * (mbab * mbca + mabb * mcba)) +
                       1. / 3 * (macb + mcab)) *
                          inverse_rho;

    // --- 6th order cumulant ---
    // C₂₂₂ = κ₂₂₂ - (4κ₁₁₁² + κ₂₀₀·κ₀₂₂ + κ₀₂₀·κ₂₀₂ + κ₀₀₂·κ₂₂₀
    //         + 4(κ₀₁₁·κ₂₁₁ + κ₁₀₁·κ₁₂₁ + κ₁₁₀·κ₁₁₂)
    //         + 2(κ₁₂₀·κ₁₀₂ + κ₂₁₀·κ₀₁₂ + κ₂₀₁·κ₀₂₁)) / ρ
    //       + (16·κ₁₁₀·κ₁₀₁·κ₀₁₁ + 4(κ₁₀₁²·κ₀₂₀ + κ₀₁₁²·κ₂₀₀ + κ₁₁₀²·κ₀₀₂)
    //         + 2·κ₂₀₀·κ₀₂₀·κ₀₀₂) / ρ²
    //       - (3(κ₀₂₂+κ₂₀₂+κ₂₂₀) + (κ₂₀₀+κ₀₂₀+κ₀₀₂)) / (9ρ)
    //       + 2(2(κ₁₀₁²+κ₀₁₁²+κ₁₁₀²) + (κ₀₀₂·κ₀₂₀+κ₀₀₂·κ₂₀₀+κ₀₂₀·κ₂₀₀)
    //         + (κ₀₀₂+κ₀₂₀+κ₂₀₀)/3) / (3ρ²)
    //       + (δρ²-δρ) / (27ρ²)                                          (Eq. J.19)
    V CUMccc =
        mccc +
        ((-4 * mbbb * mbbb - (mcaa * macc + maca * mcac + maac * mcca) -
          4 * (mabb * mcbb + mbab * mbcb + mbba * mbbc) -
          2 * (mbca * mbac + mcba * mabc + mcab * macb)) *
             inverse_rho +
         (4 * (mbab * mbab * maca + mabb * mabb * mcaa + mbba * mbba * maac) +
          2 * (mcaa * maca * maac) + 16 * mbba * mbab * mabb) *
             inverse_rho * inverse_rho -
         1. / 3 * (macc + mcac + mcca) * inverse_rho -
         1. / 9 * (mcaa + maca + maac) * inverse_rho +
         (2 * (mbab * mbab + mabb * mabb + mbba * mbba) +
          (maac * maca + maac * mcaa + maca * mcaa) +
          1. / 3 * (maac + maca + mcaa)) *
             inverse_rho * inverse_rho * 2. / 3 +
         1. / 27 * ((drho * drho - drho) * inverse_rho * inverse_rho));

    // ============================================================
    // STAGE 3: Relaxation
    //          C_αβγ → C*_αβγ (post-collision cumulants)
    //          各階 cumulant 以不同的鬆弛率趨向平衡態（平衡值均為 0）。
    //          碰撞本身不受 well-conditioning 影響。
    //          (論文 Section 4.3, Eq. 55–80)
    //
    //          程式碼 omega 對應表：
    //            omega  = ω₁  → 2階 off-diagonal (Eq. 55–57) + trace差 (Eq. 61–62)
    //            omega2 = ω₂  → 2階 trace和 (Eq. 63)
    //            omega3 = ω₃  → 3階 symmetric combinations (Eq. 64–66)
    //            omega4 = ω₄  → 3階 antisymmetric + C₁₁₁ (Eq. 67–70)
    //            omega6 = ω₆/ω₇/ω₈ → 4階 (Eq. 71–76)
    //            omega7 = ω₉  → 5階 (Eq. 77–79)
    //            omega10= ω₁₀ → 6階 (Eq. 80)
    //          目前 omega2~omega10 全設為 1（完全鬆弛至平衡態）。
    // ============================================================

    // --- 2nd order: 先做線性組合成獨立模態再鬆弛 ---
    // 將 3 個二階對角 cumulant (C₂₀₀, C₀₂₀, C₀₀₂) 重組為正交模態
    V mxxPyyPzz = mcaa + maca + maac;  // C₂₀₀ + C₀₂₀ + C₀₀₂ (trace = bulk)
    V mxxMyy    = mcaa - maca;         // C₂₀₀ - C₀₂₀         (deviatoric)
    V mxxMzz    = mcaa - maac;         // C₂₀₀ - C₀₀₂         (deviatoric)

    // 將 3 對三階 cumulant 重組為和/差模態（用於 Eq. 64–69）
    V mxxyPyzz = mcba + mabc;   // C₁₂₀ + C₁₀₂  → Eq. 64 用
    V mxxyMyzz = mcba - mabc;   // C₁₂₀ - C₁₀₂  → Eq. 67 用

    V mxxzPyyz = mcab + macb;   // C₂₁₀ + C₀₁₂  → Eq. 65 用
    V mxxzMyyz = mcab - macb;   // C₂₁₀ - C₀₁₂  → Eq. 68 用

    V mxyyPxzz = mbca + mbac;   // C₂₀₁ + C₀₂₁  → Eq. 66 用
    V mxyyMxzz = mbca - mbac;   // C₂₀₁ - C₀₂₁  → Eq. 69 用

    // (Eq. 63) C*₂₀₀ + C*₀₂₀ + C*₀₀₂ = κ₀₀₀·ω₂ + (1-ω₂)·(C₂₀₀+C₀₂₀+C₀₀₂)
    //          trace 模態用 ω₂ (bulk viscosity) 鬆弛
    mxxPyyPzz += omega2 * (maaa - mxxPyyPzz);
    // (Eq. 61) C*₂₀₀ - C*₀₂₀ = (1-ω₁)·(C₂₀₀ - C₀₂₀)
    //          deviatoric 模態用 ω₁ (shear viscosity) 鬆弛
    //          注意：省略了 Galilean correction 項 -3ρ(1-ω₁/2)(u²Dₓu - v²Dᵧv)
    //          因為 ω₂=1 時 Dₓu=Dᵧv=D_zw（見 Eq. 58–60），修正項為 0
    mxxMyy += -(-omega) * (-mxxMyy);
    // (Eq. 62) C*₂₀₀ - C*₀₀₂ = (1-ω₁)·(C₂₀₀ - C₀₀₂)
    //          同上，省略 Galilean correction
    mxxMzz += -(-omega) * (-mxxMzz);

    // (Eq. 55) C*₁₁₀ = (1-ω₁)·C₁₁₀
    mabb += omega * (-mabb);
    // (Eq. 56) C*₁₀₁ = (1-ω₁)·C₁₀₁
    mbab += omega * (-mbab);
    // (Eq. 57) C*₀₁₁ = (1-ω₁)·C₀₁₁
    mbba += omega * (-mbba);

    // --- 3rd order cumulant relaxation ---
    // (Eq. 70) C*₁₁₁ = (1-ω₅)·C₁₁₁    此處用 omega4 代替 omega5
    mbbb += omega4 * (-mbbb);
    // (Eq. 64) C*₁₂₀ + C*₁₀₂ = (1-ω₃)·(C₁₂₀ + C₁₀₂)
    mxxyPyzz += omega3 * (-mxxyPyzz);
    // (Eq. 67) C*₁₂₀ - C*₁₀₂ = (1-ω₄)·(C₁₂₀ - C₁₀₂)
    mxxyMyzz += omega4 * (-mxxyMyzz);
    // (Eq. 65) C*₂₁₀ + C*₀₁₂ = (1-ω₃)·(C₂₁₀ + C₀₁₂)
    mxxzPyyz += omega3 * (-mxxzPyyz);
    // (Eq. 68) C*₂₁₀ - C*₀₁₂ = (1-ω₄)·(C₂₁₀ - C₀₁₂)
    mxxzMyyz += omega4 * (-mxxzMyyz);
    // (Eq. 66) C*₂₀₁ + C*₀₂₁ = (1-ω₃)·(C₂₀₁ + C₀₂₁)
    mxyyPxzz += omega3 * (-mxyyPxzz);
    // (Eq. 69) C*₂₀₁ - C*₀₂₁ = (1-ω₄)·(C₂₀₁ - C₀₂₁)
    mxyyMxzz += omega4 * (-mxyyMxzz);

    // --- 2nd & 3rd order: 從獨立模態反算回個別 central moments ---
    mcaa = 1. / 3 * (mxxMyy + mxxMzz + mxxPyyPzz);
    maca = 1. / 3 * (-2 * mxxMyy + mxxMzz + mxxPyyPzz);
    maac = 1. / 3 * (mxxMyy - 2 * mxxMzz + mxxPyyPzz);

    mcba = (mxxyMyzz + mxxyPyzz) * 0.5;
    mabc = (-mxxyMyzz + mxxyPyzz) * 0.5;
    mcab = (mxxzMyyz + mxxzPyyz) * 0.5;
    macb = (-mxxzMyyz + mxxzPyyz) * 0.5;
    mbca = (mxyyMxzz + mxyyPxzz) * 0.5;
    mbac = (-mxyyMxzz + mxyyPxzz) * 0.5;

    // --- 4th order cumulant relaxation ---
    // 論文將 6 個四階 cumulant 分成兩組：
    //   同質類 C₂₂₀,C₂₀₂,C₀₂₂：重組為正交模態再各自鬆弛
    //   混合類 C₂₁₁,C₁₂₁,C₁₁₂：各自獨立鬆弛
    // 此處 ω₆ = ω₇ = ω₈ 統一為 omega6，所以不需要拆模態
    //
    // (Eq. 71) C*₂₂₀ - 2C*₂₀₂ + C*₀₂₂ = (1-ω₆)·(C₂₂₀ - 2C₂₀₂ + C₀₂₂)
    // (Eq. 72) C*₂₂₀ + C*₂₀₂ - 2C*₀₂₂ = (1-ω₆)·(C₂₂₀ + C₂₀₂ - 2C₀₂₂)
    // (Eq. 73) C*₂₂₀ + C*₂₀₂ + C*₀₂₂  = (1-ω₇)·(C₂₂₀ + C₂₀₂ + C₀₂₂)
    //          ∵ ω₆=ω₇ → 三式等價為 C*_each = (1-ω₆)·C_each
    CUMacc = (1 - omega6) * (CUMacc);   // (Eq. 71–73 合併) C*₀₂₂
    CUMcac = (1 - omega6) * (CUMcac);   // (Eq. 71–73 合併) C*₂₀₂
    CUMcca = (1 - omega6) * (CUMcca);   // (Eq. 71–73 合併) C*₂₂₀
    // (Eq. 74) C*₂₁₁ = (1-ω₈)·C₂₁₁   ∵ ω₈=ω₆ → 用 omega6
    CUMbbc = (1 - omega6) * (CUMbbc);   // (Eq. 76) C*₁₁₂ = (1-ω₈)·C₁₁₂
    CUMbcb = (1 - omega6) * (CUMbcb);   // (Eq. 75) C*₁₂₁ = (1-ω₈)·C₁₂₁
    CUMcbb = (1 - omega6) * (CUMcbb);   // (Eq. 74) C*₂₁₁ = (1-ω₈)·C₂₁₁

    // --- 5th order cumulant relaxation ---
    // (Eq. 79) C*₁₂₂ = (1-ω₉)·C₁₂₂   此處 omega7 對應論文 ω₉
    CUMbcc += omega7 * (-CUMbcc);
    // (Eq. 78) C*₂₁₂ = (1-ω₉)·C₂₁₂
    CUMcbc += omega7 * (-CUMcbc);
    // (Eq. 77) C*₂₂₁ = (1-ω₉)·C₂₂₁
    CUMccb += omega7 * (-CUMccb);

    // --- 6th order cumulant relaxation ---
    // (Eq. 80) C*₂₂₂ = (1-ω₁₀)·C₂₂₂
    CUMccc += omega10 * (-CUMccc);

    // ============================================================
    // STAGE 4: Backward Cumulants Transformation
    //          C*_αβγ (post-collision cumulants) → κ*_αβγ (central moments)
    //          將鬆弛後的 cumulants 加回低階乘積項，
    //          還原成 post-collision central moments。
    //          這是 Stage 2 的逆運算（解 Eq. J.16–J.19 求 κ*）。
    //          (論文 Section 4.4, Eq. 81–84；
    //           well-conditioned 版本見 Appendix J, Eq. J.16–J.19 逆算)
    // ============================================================
    // --- 4th order: cumulants → central moments ---
    // (Eq. 81 / J.16 逆算)
    // κ*₂₁₁ = C*₂₁₁ + ((κ*₂₀₀ + 1/3)·κ*₀₁₁ + 2·κ*₁₁₀·κ*₁₀₁)/ρ
    mcbb = CUMcbb +
           1. / 3 * ((3 * mcaa + 1) * mabb + 6 * mbba * mbab) * inverse_rho;
    // (Eq. 81 permuted) κ*₁₂₁
    mbcb = CUMbcb +
           1. / 3 * ((3 * maca + 1) * mbab + 6 * mbba * mabb) * inverse_rho;
    // (Eq. 81 permuted) κ*₁₁₂
    mbbc = CUMbbc +
           1. / 3 * ((3 * maac + 1) * mbba + 6 * mbab * mabb) * inverse_rho;

    // (Eq. 82 / J.17 逆算)
    // κ*₂₂₀ = C*₂₂₀ + ((κ*₂₀₀·κ*₀₂₀ + 2·κ*₁₁₀²) + (κ*₂₀₀+κ*₀₂₀)/3)/ρ - δρ/(9ρ)
    mcca = CUMcca + (((mcaa * maca + 2 * mbba * mbba) * 9 + 3 * (mcaa + maca)) *
                         inverse_rho -
                     (drho * inverse_rho)) *
                        1. / 9;
    // (Eq. 82 permuted) κ*₂₀₂
    mcac = CUMcac + (((mcaa * maac + 2 * mbab * mbab) * 9 + 3 * (mcaa + maac)) *
                         inverse_rho -
                     (drho * inverse_rho)) *
                        1. / 9;
    // (Eq. 82 permuted) κ*₀₂₂
    macc = CUMacc + (((maac * maca + 2 * mabb * mabb) * 9 + 3 * (maac + maca)) *
                         inverse_rho -
                     (drho * inverse_rho)) *
                        1. / 9;

    // --- 5th order: cumulants → central moments ---
    // (Eq. 83 / J.18 逆算)
    // κ*₁₂₂ = C*₁₂₂ + ((κ₀₀₂·κ₁₂₀ + κ₀₂₀·κ₁₀₂ + 4κ₀₁₁·κ₁₁₁
    //         + 2(κ₁₀₁·κ₀₂₁ + κ₁₁₀·κ₀₁₂)) + (κ₁₂₀+κ₁₀₂)/3) / ρ
    mbcc = CUMbcc + 1. / 3 *
                        (3 * (maac * mbca + maca * mbac + 4 * mabb * mbbb +
                              2 * (mbab * macb + mbba * mabc)) +
                         (mbca + mbac)) *
                        inverse_rho;
    // (Eq. 83 permuted) κ*₂₁₂
    mcbc = CUMcbc + 1. / 3 *
                        (3 * (maac * mcba + mcaa * mabc + 4 * mbab * mbbb +
                              2 * (mabb * mcab + mbba * mbac)) +
                         (mcba + mabc)) *
                        inverse_rho;
    // (Eq. 83 permuted) κ*₂₂₁
    mccb = CUMccb + 1. / 3 *
                        (3 * (mcaa * macb + maca * mcab + 4 * mbba * mbbb +
                              2 * (mbab * mbca + mabb * mcba)) +
                         (macb + mcab)) *
                        inverse_rho;

    // --- 6th order: cumulant → central moment ---
    // (Eq. 84 / J.19 逆算)
    // κ*₂₂₂ = C*₂₂₂ + (和 J.19 相同的乘積項，但符號反轉)
    mccc =
        CUMccc -
        ((-4 * mbbb * mbbb - (mcaa * macc + maca * mcac + maac * mcca) -
          4 * (mabb * mcbb + mbab * mbcb + mbba * mbbc) -
          2 * (mbca * mbac + mcba * mabc + mcab * macb)) *
             inverse_rho +
         (4 * (mbab * mbab * maca + mabb * mabb * mcaa + mbba * mbba * maac) +
          2 * (mcaa * maca * maac) + 16 * mbba * mbab * mabb) *
             inverse_rho * inverse_rho -
         1. / 9 * (macc + mcac + mcca) * inverse_rho -
         1. / 9 * (mcaa + maca + maac) * inverse_rho +
         (2 * (mbab * mbab + mabb * mabb + mbba * mbba) +
          (maac * maca + maac * mcaa + maca * mcaa) +
          1. / 3 * (maac + maca + mcaa)) *
             inverse_rho * inverse_rho * 2. / 3 +
         1. / 27 * ((drho * drho - drho) * inverse_rho * inverse_rho));

    // ============================================================
    // STAGE 5: Backward Central Moment Transformation
    //          κ*_αβγ (central moments) → f̄*_ijk (well-conditioned populations)
    //          使用反向 Chimera 變換，沿 x→y→z 依序將
    //          post-collision central moments 轉回 populations。
    //          (論文 Appendix J, Eq. J.20–J.28；
    //           原始版本見 Section 4.5, Eq. 88–96)
    // ============================================================
    // body force 修正（論文 Section 4.4, Eq. 85–87）
    // 在 Chimera 變換前，半力已施加於參考系速度 u 的定義中。
    // 碰撞後再反轉一階 moments，等效於在碰撞前後各施加半力，
    // 確保時間二階精度（time-symmetric splitting）。
    // (Eq. 85) κ*₁₀₀ = -κ₁₀₀
    mbaa = -mbaa;
    // (Eq. 86) κ*₀₁₀ = -κ₀₁₀
    maba = -maba;
    // (Eq. 87) κ*₀₀₁ = -κ₀₀₁
    maab = -maab;

    // 反向 Chimera 變換 (Eq. J.20–J.28)
    computePopulations(moments, cell, u);

    return uSqr;
  }
};













































namespace collision {
  struct CUM {
  using parameters = typename meta::list<descriptors::OMEGA>;

  static std::string getName() {
    return "CUM";
  }

  template <typename DESCRIPTOR, typename MOMENTA, typename EQUILIBRIUM>
  struct type {

    static_assert(DESCRIPTOR::d==3 && DESCRIPTOR::q == 27, "Cumulant Dynamics only implemented in D3Q27");
    using MomentaF = typename MOMENTA::template type<DESCRIPTOR>;
    using EquilibriumF = typename EQUILIBRIUM::template type<DESCRIPTOR,MOMENTA>;

    template <typename CELL, typename PARAMETERS, typename V=typename CELL::value_t>
    CellStatistic<V> apply(CELL& cell, PARAMETERS& parameters) any_platform {
      const V omega = parameters.template get<descriptors::OMEGA>();
      V rho,       // 四階 cumulants（例如 CUMcbb = C₂₁₁）
      V CUMcbb = mcbb - ((mcaa + 1./3) * mabb + 2*mbba*mbab) * inverse_rho;
      //       = κ₂₁₁ - (1/ρ)[(κ₂₀₀ + 1/3)·κ₀₁₁ + 2·κ₁₁₀·κ₁₀₁]
      
      // 四階 cumulants（例如 CUMcca = C₂₂₀）
      V CUMcca = mcca - (((mcaa*maca + 2*mbba*mbba) + 1./3*(mcaa+maca)) * inverse_rho
                 - 1./9*(drho*inverse_rho));
      //       = κ₂₂₀ - (1/ρ)[κ₂₀₀·κ₀₂₀ + 2κ₁₁₀² + (1/3)(κ₂₀₀+κ₀₂₀)] + (1/9)(δρ/ρ)u[DESCRIPTOR::d];
      MomentaF().computeRhoU(cell, rho, u);
      V uSqr = cum<DESCRIPTOR>::cumCollision(cell, omega, rho, u);
      return {rho, uSqr};
    };
  };
};
}
}


#endif
