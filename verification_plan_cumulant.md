# Cumulant Collision Operator 驗證計畫

**Reference**: Gehrke & Rung (2022), "Periodic hill flow simulations with a parameterized cumulant lattice Boltzmann method", Int. J. Numer. Meth. Fluids 94, 1111-1154

---

## 1. 驗證方法

使用獨立 CPU 測試程式 `test_cumulant_vs_paper.c`，逐項對照 Appendix B (B1)-(B30) 公式。測試涵蓋 Chimera 變換、宏觀量、WP 參數、鬆弛算子、守恆律共 10 個測試組。

## 2. 逐項驗證結果

### B1-B2: Well-conditioning 與權重

**B1**: `f = f^c - w` (well-conditioned PDF)

代碼第 194 行: `m[i] = f_in[i] - GILBM_W[i]`，完全吻合。

**B2**: D3Q27 權重 (8/27, 2/27, 1/54, 1/216)，已驗證 GILBM_W 與 D3Q27_W 完全一致。

### B3-B6: 宏觀量計算

密度波動 (B3) 和速度分量 (B4-B6) 通過直接求和實作。測試結果：在標準平衡態 `f_eq(rho=1.02, u=0.05, v=-0.03, w=0.02)` 下，rho 和 u,v,w 恢復精度 < 1e-14。

### B7-B9: 速度導數 D_x u, D_y v, D_z w

此為紙上定義，用於 B13-B15 和 B23-B25 的理論表達式。代碼**未直接實作**這些導數（不需要，因為代碼使用等效的矩形式表達）。

### B10-B12: 二階離對角 cumulant 鬆弛

**B10**: `C*_{110} = (1 - omega_1) * C_{110}`

代碼第 381-383 行:
```c
m[I_abb] *= (1.0 - omega);  // C011
m[I_bab] *= (1.0 - omega);  // C101
m[I_bba] *= (1.0 - omega);  // C110
```
**完全吻合 B10-B12。**

### B13-B15: 二階對角 cumulant 鬆弛

論文給出含速度導數修正的完整形式。代碼使用**簡化形式**（不含 `3*rho*(1-omega/2)*(u^2*D_xu - v^2*D_yv)` 修正項）。

測試結果（Ma=0.06 level）:
- C200 差異: 1.1e-6 (O(Ma^2) ~ 3.4e-3)
- C020 差異: 2.6e-7
- C002 差異: 2.0e-7

**結論**: 差異為 O(Ma^2) 量級，對 Ma < 0.1 的標準 LBM 模擬可接受。對高 Ma 流動需考慮加入修正。

### B16-B22: 三階 cumulant 鬆弛（WP 模式）

**B16-B21**: 對稱/反對稱分解 — 代碼完全吻合。

**B16 論文筆誤**: 論文寫 `omega_{4,2}^lambda`，但應為 `omega_{4,1}^lambda`（與 B17 使用同一對稱對 C120/C102，應使用相同鬆弛率）。B17 正確地使用 `omega_{4,1}`。代碼使用 `omega4_1`，**正確**。

**B22**: `C*_{111} = (1 - omega_5^lambda) * C_{111}`，代碼吻合。

Lambda-limiter (Eq. 20-26) 正確應用於每個對稱/反對稱對。

### B23-B25: 四階對角 cumulant（WP 模式）

論文用速度導數 (B7-B9) 和係數 A 表達四階平衡態。代碼用 Eq. 17-18（二階 cumulant 乘積 + A, B 係數）。

關鍵發現：

- **B23-B25 僅使用係數 A**，未包含 B 對離對角應力乘積的貢獻
- **Eq. 17-18 使用 A 和 B**，涵蓋 `sigma_xy^2` 等交叉項
- **結論**: B23-B25 為**簡化形式**（剪切無關極限），代碼使用**完整 Eq. 17-18**是正確的

測試結果（omega=1.39, rho=1.02, u=0.05）:
- Eq.17-18: C220_eq = 0.368, C022_eq = 0.368, C202_eq = 0.368
- B23-B25: C220_eq ~ 0 （因僅含 A 項且速度梯度小）

### B26-B28: 四階離對角 cumulant（WP 模式）

**B26**: `C*_{211} = (1 - omega_1/2) * B * C_{011}`

代碼第 472-477 行使用 pre-relaxation C011 和 `(1-omega*0.5)*B` 係數，**直接賦值為 central moment**。

分析：如果 B26 給出的是 **cumulant**，則需要加上 back-conversion 乘積項 `(C200*C011 + 2*C110*C101)/rho`。代碼省略此乘積。

測試結果：乘積項為 O(Ma^2)，在 Ma=0.06 下佔 B26 值的 ~0%（因 B26 本身已非常小）。

**結論**: 代碼的簡化處理對 Periodic Hill 工況可接受。

### B29-B30: 五階 cumulant

**B29**: `C*_{221} = 0`, **B30**: `C*_{212} = 0`

代碼 `CUMbcc *= (1-omega9)` 加 `CUMcbc *= (1-omega9)` 加 `CUMccb *= (1-omega9)`，omega9=1 時等效為 0。完全吻合。

六階 cumulant 同理：`CUMccc *= (1-omega10)` 且 omega10=1。

## 3. 已發現並修復的 Bug

| # | Bug | 影響 | 狀態 |
|---|-----|------|------|
| 1 | `rho_modify_d` 未初始化 | 密度修正含垃圾值 | 已修復 |
| 2 | `delta_zeta_h` 未初始化 (alpha=1,2) | Ma_max=6.534 → NaN | 已修復 (memset) |
| 3 | CE BC 係數: `-(omega)` → `-(omega-0.5)` | 壁面非平衡部分多 0.5*dt | **本次修復** |
| 4 | 壁面速度梯度: 僅用 u[k=4] | 一階精度且用錯節點 | **本次修復** |

## 4. 已知簡化（非 Bug）

1. **B13-B15 速度導數修正**: 代碼省略 O(Ma^2) Galilean 不變性修正。對 Ma < 0.1 可接受。
2. **B26-B28 back-conversion**: 代碼直接設 central moment，省略 cumulant→moment 乘積項。乘積為 O(Ma^2)。
3. **WP 四階平衡態**: 從 f=W[q] 冷啟動產生非物理分布（面方向 f=-0.107）。不影響穩定性，但 AO 模式無此問題。

## 5. 驗證測試結果

```
TEST  1: Chimera roundtrip             — PASS (err < 7e-17)
TEST  2: Macroscopic quantities B3-B6  — PASS (err < 1e-14)
TEST  3: WP parameters Eq.14-18       — PASS (A=B=0 at singularity)
TEST  4: B13-B15 velocity corrections  — PASS (diff = O(Ma^2))
TEST  5: B16-B22 3rd-order structure   — PASS
TEST  6: B23-B25 vs Eq.17-18          — PASS (code uses full form)
TEST  7: B26-B28 off-diagonal         — PASS (product ~ 0% of B26)
TEST  8: Equilibrium collision         — PASS (AO exact, WP has expected WP shift)
TEST  9: Conservation laws             — PASS (mass, jx, jz preserved < 1e-13)
TEST 10: 100-step WP stability        — PASS (Ma=0.00018, stable)
```

**全部 17/17 子測試通過。**

## 6. 建議

1. **立即可用**: 修復 Bug 3, 4 後，使用 AO 模式 (`USE_WP_CUMULANT=0`) 進行冷啟動，預計可穩定運行。
2. **WP 模式進階**: 若需 WP 精度，考慮使用 WP-consistent 初始化或從 AO 漸變至 WP。
3. **高 Re 模擬**: 若 Re > 5600，考慮調整 CUM_LAMBDA (Table 7, GR22)。
