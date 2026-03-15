# 方案 A：Matrix-based Moment Transform 理論基礎

## 1. 問題：Chimera 遞迴掃描放大 GILBM 插值噪聲

原始 Chimera 變換將**固定線性運算**（raw moment 計算）與**速度相關運算**（binomial shift）合併為單一遞迴掃描。每個 triplet 的輸出立即被下一步使用：

```
m[a] = fa + fb + fc           ← raw m0（已更新）
m[b] = diff - (m[a] + K)·u   ← 使用剛計算的 m[a]，乘以 u
m[c] = sum - 2·diff·u + (m[a]+K)·u²  ← 再次乘以 u²
```

在 GILBM 框架中，7 點 Lagrange 插值會在分佈函數 f 中引入振盪噪聲 δf。Chimera 的遞迴結構讓速度相關項（u·δf, u²·δf）直接作用在帶噪聲的**個別** f 值上，導致誤差隨掃描累積放大。

## 2. 解決方案：分離為兩階段

### Phase 1：固定矩陣 M_1d（Raw Moment Transform）

```
        ⎡ 1   1   1 ⎤   ⎡ f_minus ⎤   ⎡ m0 ⎤
M_1d =  ⎢-1   0   1 ⎥ · ⎢ f_zero  ⎥ = ⎢ m1 ⎥
        ⎣ 1   0   1 ⎦   ⎣ f_plus  ⎦   ⎣ m2 ⎦
```

- **m0 = f₋ + f₀ + f₊**（加總：噪聲被平均）
- **m1 = f₊ − f₋**（差分：對稱噪聲抵消）
- **m2 = f₋ + f₊**（部分和）
- det(M_1d) = 2，條件數良好
- **完全不使用速度 u** → 噪聲通過線性組合被抑制

### Phase 2：Binomial Shift S(u)

```
        ⎡ 1    0    0 ⎤   ⎡ m0 ⎤   ⎡ κ0 ⎤
S(u) =  ⎢-u    1    0 ⎥ · ⎢ m1 ⎥ = ⎢ κ1 ⎥
        ⎣ u²  -2u   1 ⎦   ⎣ m2 ⎦   ⎣ κ2 ⎦
```

- **κ0 = m0**（不變）
- **κ1 = m1 − (m0 + K)·u**（中心矩）
- **κ2 = m2 − 2·m1·u + (m0 + K)·u²**（中心矩）
- u 相關運算作用在**已聚合的矩量**上，而非個別 f 值
- K = well-conditioning 常數（與 Chimera 相同）

### 關鍵差異

| 項目 | Chimera（原始） | Option A（矩陣） |
|------|----------------|------------------|
| u 乘法對象 | 個別 f 值（含噪聲） | 聚合後的矩量（噪聲已平均） |
| 遞迴耦合 | m[b], m[c] 使用剛更新的 m[a] | Phase 1 與 Phase 2 完全獨立 |
| 數學結果 | 中心矩 κ | 完全相同的中心矩 κ |
| 噪聲放大 | u·δf 逐步累積 | δf 先被線性組合抑制 |

## 3. 數學等價性證明

### 3.1 張量積結構

27×27 變換矩陣為 **M = M_z ⊗ M_y ⊗ M_x**，其中每個 3×3 子矩陣可獨立作用。完整的 forward transform 為：

```
T = S(u) · M_1d   （先 M_1d 再 S(u)，逐方向掃描）
```

Chimera 將 T 合併為單一遞迴步驟。Option A 將 T 拆分為兩個獨立循環。由於 M_1d 不含 u，兩種實現產生完全相同的結果。
<!--  -->
### 3.2 逆變換

```
逆變換 = M⁻¹_1d · S⁻¹(u)   （先逆 shift 再逆矩陣）
```

**S⁻¹(u):**
```
          ⎡ 1    0    0 ⎤
S⁻¹(u) = ⎢ u    1    0 ⎥
          ⎣ u²   2u   1 ⎦
```

**M⁻¹_1d = (1/2):**
```
            ⎡ 0  -1   1 ⎤
(1/2) ·     ⎢ 2   0  -2 ⎥
            ⎣ 0   1   1 ⎦
```

- f₋ = (−m1 + m2) / 2
- f₀ = m0 − m2
- f₊ = (m1 + m2) / 2

### 3.3 Round-trip 驗證

```
M⁻¹_1d · S⁻¹(u) · S(u) · M_1d = M⁻¹_1d · I · M_1d = I  ✓
```

## 4. Well-conditioning 常數 K 的角色

K[p] 定義為：在 u=0 時對權重 W[27] 施加 forward Chimera 後，各 triplet 的第零矩（sum）。

在 Phase 2 中：
- κ1 = m1 − **(m0 + K)**·u
- κ2 = m2 − 2·m1·u + **(m0 + K)**·u²

K 加在 m0 上，補償 f−W 的 well-conditioning 偏移。逆變換中 K 以相同方式出現，確保正確抵消。

## 5. CUM_IDX 映射

CUM_IDX[27][3] 定義 27 個 triplet 的索引映射，分為三組掃描：

- Passes 0−8: z 方向（按 (ex,ey) 分組，掃描 ez = {−1, 0, +1}）
- Passes 9−17: y 方向（按 (ex, z-order) 分組，掃描 ey）
- Passes 18−26: x 方向（按 (y-order, z-order) 分組，掃描 ex）

Option A 使用完全相同的 CUM_IDX 和 CUM_K，只改變每個 triplet 內的運算方式。

## 6. 預期效果

1. **消除 GILBM 噪聲放大**：Phase 1 的線性組合在速度乘法之前平均噪聲
2. **保持物理精度**：數學結果與 Chimera 完全一致
3. **無額外人工黏性**：不像濾波（方案 C），不會降低有效雷諾數
4. **可達目標 Re = 700**：消除了 Δν/ν = nσdt/(6ν) 的黏性損失

## 7. 參考文獻

- Geier et al., Comp. Math. Appl. 70(4), 507-547, 2015 — Chimera transform 原始推導
- Geier et al., J. Comput. Phys. 348, 862-888, 2017 — WP 參數化
- Imamura et al., 2005 — GILBM 框架與逆變速度
- Gehrke & Rung, Int. J. Numer. Meth. Fluids 94, 1111-1154, 2022 — Lambda limiter
