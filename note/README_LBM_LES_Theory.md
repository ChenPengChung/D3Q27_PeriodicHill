# LBM-LES 理論基礎與 Smagorinsky 亞格子模型

## 目錄
1. [概述](#概述)
2. [濾波理論基礎](#濾波理論基礎)
3. [LBM 中的隱式濾波](#lbm-中的隱式濾波)
4. [Smagorinsky 亞格子模型](#smagorinsky-亞格子模型)
5. [實作細節](#實作細節)
6. [常見問答 Q&A](#常見問答-qa)

---

## 概述

本專案使用 D3Q19 MRT Lattice Boltzmann Method 搭配 Smagorinsky 大渦模擬 (LES) 亞格子模型，用於模擬週期山丘流場在高雷諾數 (Re = 700) 下的湍流行為。

### 動機

在高雷諾數下，基礎鬆弛時間 τ 接近穩定性極限：
- τ = 0.6833 接近 0.5
- 網格解析度不足以解析所有湍流尺度
- 需要 LES 亞格子模型補償未解析尺度

---

## 濾波理論基礎

### 傳統 LES (Navier-Stokes)

對速度場應用空間濾波：

$$\bar{\phi}(\mathbf{x}, t) = \int_{-\infty}^{\infty} G(\mathbf{x} - \mathbf{x}')\phi(\mathbf{x}', t) d\mathbf{x}'$$

其中 $G(\mathbf{x})$ 為濾波核函數，滿足：

$$\int_{-\infty}^{\infty} G(\mathbf{x}) d\mathbf{x} = 1$$

### 常見濾波核函數

| 濾波器類型 | 核函數 $G(\mathbf{x})$ | 特性 |
|------------|------------------------|------|
| **Box Filter** | $\begin{cases} 1/\Delta^3 & \|\mathbf{x}\| \leq \Delta/2 \\ 0 & \text{otherwise} \end{cases}$ | 物理空間局部平均 |
| Gaussian | $\left(\frac{6}{\pi\Delta^2}\right)^{3/2} \exp\left(-\frac{6\|\mathbf{x}\|^2}{\Delta^2}\right)$ | 平滑衰減 |
| Sharp Spectral | $\begin{cases} 1 & \|k\| \leq \pi/\Delta \\ 0 & \text{otherwise} \end{cases}$ | 波數空間截斷 |

---

## LBM 中的隱式濾波

### 核心概念

在 LBM 中，**不進行顯式濾波操作**。濾波是由網格離散化**隱式**完成的。

### 為什麼是 Box Filter？

```
     |----Δx----|
     ●─────────●─────────●
    x_{i-1}    x_i     x_{i+1}
         ↑
    這個格點代表
    [x_i - Δx/2, x_i + Δx/2] 
    區間內的平均值
```

1. **格點離散化**：每個格點儲存一個值，代表周圍體積 $\Delta x \times \Delta y \times \Delta z$ 的平均
2. **有限體積觀點**：$f_i(\mathbf{x}) = \frac{1}{\Delta^3} \int_{V_{cell}} f_i(\mathbf{x}') d\mathbf{x}'$
3. **這正是 Box Filter 的定義**

### LBM-LES 的濾波假設

| 項目 | 設定 |
|------|------|
| **濾波核函數** | Box Filter（隱式，由網格離散化決定） |
| **濾波寬度** | Δ = 1 格子單位 |
| **顯式卷積** | 無（隱式處理） |

### 濾波後的 Lattice Boltzmann Equation

求解的是濾波後的分布函數方程：

$$\bar{f}_i(\mathbf{x} + \mathbf{e}_i \Delta t, t + \Delta t) - \bar{f}_i(\mathbf{x}, t) = \bar{\Omega}_i$$

其中：
- $\bar{f}_i$ = 濾波後的分布函數
- $\bar{\Omega}_i$ = 濾波後的碰撞算子

巨觀量：
- $\bar{\rho} = \sum_i \bar{f}_i$ （濾波後密度）
- $\bar{\mathbf{u}} = \sum_i \mathbf{e}_i \bar{f}_i / \bar{\rho}$ （濾波後速度）

---

## Smagorinsky 亞格子模型

### 亞格子應力

濾波後的 Navier-Stokes 方程產生亞格子應力項：

$$\tau_{ij}^{sgs} = \overline{u_i u_j} - \bar{u}_i \bar{u}_j$$

這項無法直接計算（需要未濾波速度）。

### Smagorinsky 模型

假設亞格子應力正比於濾波應變率：

$$\tau_{ij}^{sgs} - \frac{1}{3}\tau_{kk}\delta_{ij} = -2\nu_t \bar{S}_{ij}$$

渦黏性係數：

$$\nu_t = (C_s \Delta)^2 |\bar{S}|$$

其中：
- $C_s$ = Smagorinsky 常數 (0.1 ~ 0.2)
- $\Delta$ = 濾波寬度 (= 1 格子單位)
- $|\bar{S}| = \sqrt{2\bar{S}_{ij}\bar{S}_{ij}}$ = 應變率幅值

### LBM 中的實作

#### Step 1: 非平衡應力張量

$$\Pi_{\alpha\beta} = \sum_{i} e_{i\alpha} e_{i\beta} (f_i - f_i^{eq})$$

#### Step 2: 二階不變量

$$Q = \Pi_{\alpha\beta} \Pi_{\alpha\beta}$$

#### Step 3: 應變率幅值 (解析解)

$$|\bar{S}| = \frac{\sqrt{\nu_0^2 + 18 C_s^2 \Delta^2 \sqrt{Q}} - \nu_0}{6 C_s^2 \Delta^2}$$

#### Step 4: 總黏性與鬆弛時間

$$\nu_{total} = \nu_0 + (C_s \Delta)^2 |\bar{S}|$$

$$\tau_{total} = 3 \frac{\nu_{total}}{dt} + 0.5$$

---

## 實作細節

### 配置參數 (`variables.h`)

```cpp
#define SMAGORINSKY  1          // 1=啟用 LES, 0=停用
#define C_Smag       0.2        // Smagorinsky 常數
#define DELTA        (1.0)      // 濾波寬度 (格子單位)
```

### 演算法流程

```
1. Stream: 從鄰近節點收集 f_in
2. 計算巨觀量 (ρ, u, v, w)
3. 計算平衡分布 f_eq
4. [LES] 計算 f_neq = f_in - f_eq
5. [LES] 計算應力張量 Π_αβ
6. [LES] 計算 Q = Π_αβ Π_αβ
7. [LES] 計算 |S̄|
8. [LES] 更新 τ_total, s9, s11, s13, s14, s15
9. 執行 MRT 碰撞
10. 儲存 f_new
```

---

## 常見問答 Q&A

### Q1: 我需要顯式計算濾波卷積嗎？

**不需要。** LBM 的離散化本身隱式完成了濾波。你直接從 $f_i$ 計算的 $\rho$ 和 $\mathbf{u}$ 已經被視為濾波後的量。

---

### Q2: 濾波核函數是什麼？

**Box Filter（盒式濾波器）。** 這是由 LBM 的網格離散化隱式決定的，不是你選擇的：

$$G(\mathbf{x}) = \begin{cases} 1/\Delta^3 & |\mathbf{x}| \leq \Delta/2 \\ 0 & \text{otherwise} \end{cases}$$

每個格點的值代表該格子體積內的平均值。

---

### Q3: 濾波寬度 Δ 是怎麼決定的？

**由網格間距決定。** 在格子單位下：

$$\Delta = \Delta x = \Delta y = \Delta z = 1$$

這就是為什麼 `DELTA = 1.0`。

---

### Q4: 我怎麼知道分布函數是濾波後的？

**你無法從 $f_i$ 本身區分。** 判斷方法是**網格解析度**：

| 條件 | 模式 | $f_i$ 的意義 |
|------|------|--------------|
| $\Delta x \ll \eta$ (Kolmogorov 尺度) | DNS | 真實分布函數 |
| $\Delta x > \eta$ | **LES** | 濾波後分布函數 |

對於 Re = 700，Kolmogorov 尺度 $\eta \ll \Delta x$，無法解析 → 必須視為 LES。

**開啟 Smagorinsky 的那一刻，就是在聲明 $f_i = \bar{f}_i$。**

---

### Q5: 真實的分布函數可以求出嗎？

**不可以。** 真實分布函數包含所有尺度的資訊，被濾波「抹平」了。這就是 LES 的本質：

```
真實分布函數 f         濾波變換         濾波後分布函數 f̄
(包含所有尺度)   ──────────────>    (你計算的)
     ↑                                    ↓
  不可解析                            用 Smagorinsky
  無法求出                            補償丟失資訊
```

---

### Q6: Smagorinsky 模型在補償什麼？

**能量耗散。** 小尺度渦旋的效應是把能量從大尺度傳遞到小尺度，最終被分子黏性耗散。由於無法解析小尺度，Smagorinsky 用增加的渦黏性 $\nu_t$ 直接耗散這些能量：

$$\underbrace{\text{大尺度}}_{\text{解析}} \xrightarrow{\text{能量級聯}} \underbrace{\text{小尺度}}_{\text{未解析}} \xrightarrow{\nu_t} \text{耗散}$$

---

### Q7: 為什麼我不需要選擇濾波核？

因為 **LBM 的離散格點表示法自動隱含了 Box Filter**。這是所有有限差分/有限體積方法的共同特性。你無法改變它，除非使用顯式濾波（但這會增加計算成本且不常見）。

---

### Q8: C_Smag 該設多少？

| $C_s$ 值 | 效果 |
|----------|------|
| 0.10 | 標準值，最小耗散 |
| 0.15 | 中等耗散，較好穩定性 |
| 0.20 | 高耗散，最大穩定性 |

**建議**：從 0.1 開始，如果發散則逐步增加到 0.15、0.2。

---

### Q9: 限制外力大小可以解決發散嗎？

**不建議。** 限制外力會導致：
- Ub_avg 永遠無法達到 Uref
- 實際 Re 比設定值低
- 這是非物理的做法

**物理正確的解法**：
1. 增加網格解析度 (推薦)
2. 降低目標 Re
3. 用定壓差驅動 (接受較低的實際 Re)

---

### Q10: 為什麼增加 NZ 可以提高穩定性？

$$dt \propto \frac{1}{NZ} \quad \Rightarrow \quad Uref = Re \cdot \nu \propto \frac{1}{NZ}$$

增加 NZ：
- dt 減小 → 流速降低
- 遠離 Ma ≈ 1 的不穩定區域
- 局部 τ 值更安全

| 設定 | NZ=64 | NZ=128 |
|------|-------|--------|
| dt | ~0.019 | ~0.0095 |
| Uref | ~0.816 | ~0.408 |
| 穩定性 | 差 | 好 |
| 計算成本 | 1x | 2x |

---

## 總結

### LBM-LES 的核心假設

1. **隱式濾波**：網格離散化 = Box Filter，Δ = 1 格子單位
2. **濾波後求解**：$f_i$、$\rho$、$\mathbf{u}$ 都是濾波後的量
3. **亞格子閉合**：Smagorinsky 渦黏性補償未解析尺度

### 你的設定

| 參數 | 值 | 說明 |
|------|-----|------|
| 濾波核 | Box Filter | 隱式決定 |
| Δ | 1.0 | 格子單位 |
| C_s | 0.2 | Smagorinsky 常數 |
| Re | 700 | 目標雷諾數 |
| 網格 | 32×128×128 | NZ 增倍以提高穩定性 |

---

## 參考文獻

1. Smagorinsky, J. (1963). General circulation experiments with the primitive equations. *Monthly Weather Review*, 91(3), 99-164.

2. Hou, S., Sterling, J., Chen, S., & Doolen, G. D. (1996). A lattice Boltzmann subgrid model for high Reynolds number flows. *Pattern Formation and Lattice Gas Automata*, 6, 149-157.

3. Yu, H., Girimaji, S. S., & Luo, L. S. (2005). DNS and LES of decaying isotropic turbulence with and without frame rotation using lattice Boltzmann method. *Journal of Computational Physics*, 209(2), 599-616.
