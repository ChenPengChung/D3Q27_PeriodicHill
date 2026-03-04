# FTT-Gated Two-Stage Time Averaging — 六項交叉檢查報告

> Date: 2026-03-02
> Branch: `Edit3_GILBM`
> 修改涵蓋 6 個檔案: `variables.h`, `memory.h`, `statistics.h`, `evolution.h`, `main.cu`, `fileIO.h`

---

## 修改摘要

實作 FTT-gated 兩階段時間平均架構：

- **Stage 1** (FTT ≥ 20): 累積平均速度 (`u_tavg`, `v_tavg`, `w_tavg`)，計數器 `vel_avg_count`
- **Stage 2** (FTT ≥ 50): 累積 Reynolds stress (MeanVars 24 陣列 + MeanDerivatives 9 陣列)，計數器 `rey_avg_count`
- **Bug fixes**: P1 (cudaMemset TBSWITCH), P2 (Ub_avg_d memset), L3 (MeanDerivatives 使用 dk_dz/dk_dy 度量項取代未初始化的 ZSlopePara_d)
- **Instantaneous Ub**: 移除累積式 Ub_avg，改為每次 Launch_ModifyForcingTerm 單次 GPU 計算
- **VTK 命名**: benchmark 慣例 (U_mean, W_mean, V_mean, uu, ww, vv, uw, k)
- **Binary checkpoint**: 每 1000 步寫入 32 個統計陣列 + `rey_avg_count`
- **Backward compat**: 舊 VTK (tavg_count=, v_time_avg, w_time_avg) 仍可解析

---

## 1. 編譯層面 — PASS

### 新增變數宣告完整性

| 新變數 | 宣告位置 | 使用位置 | 狀態 |
|--------|---------|---------|------|
| `FTT_STAGE1` | variables.h:83 `#define` | main.cu:576,584,590,606,613,629,631 | PASS |
| `FTT_STAGE2` | variables.h:84 `#define` | main.cu:576,594,606,631 | PASS |
| `FINAL_STATS_VTK` | variables.h:87 `#define` | main.cu:764 `#if` | PASS |
| `vel_avg_count` | main.cu:94 `int` | fileIO.h:674,818,848,886,930; main.cu:469,586,615,632,758 | PASS |
| `rey_avg_count` | main.cu:95 `int` | fileIO.h:234,277,693,717,826,853,886,972; main.cu:495,578,608,632,679,752,754,759 | PASS |
| `stage1_announced` | main.cu:96 `bool` | main.cu:485,590-591 | PASS |
| `stage2_announced` | main.cu:97 `bool` | main.cu:497,594-595 | PASS |
| `u_tavg_h` | main.cu:92 `double*` | main.cu:173,472-475,652,772; fileIO.h:418,453,684 | PASS |
| `u_tavg_d` | main.cu:93 `double*` | memory.h:139,142,209; evolution.h:78; main.cu:479 | PASS |

### 函數簽名匹配

| 函數 | 宣告參數數 | 呼叫參數數 | 狀態 |
|------|-----------|-----------|------|
| `AccumulateTavg_Kernel` (evolution.h:64) | 7 | 7 (evolution.h:78) | PASS |
| `MeanDerivatives` (statistics.h:181) | 16 | 16 (statistics.h:253) | PASS |
| `MeanVars` (statistics.h:122) | 28 | 28 (statistics.h:245) | PASS |
| `Launch_AccumulateTavg` (evolution.h:74) | 0 | 0 (main.cu:585,614) | PASS |
| `Launch_TurbulentSum` (statistics.h:241) | 1 | 1 (main.cu:577,607) | PASS |

### 已移除變數確認

- `ZSlopePara_d[5]`: main.cu:39 只剩 comment，memory.h 無分配/釋放 → PASS
- `ub_accu_count`: main.cu:104 只剩 comment，無活躍引用 → PASS
- `time_avg_count`: 無任何引用 → PASS
- `DISS`: 僅在 statistics.h 被註解區塊 (L4-120) 中 → PASS

### Include 順序

main.cu: statistics.h (L144) → evolution.h (L145) → fileIO.h (L146)，所有全域變數在 include 之前宣告 → PASS

---

## 2. 計數器一致性 — PASS

### `vel_avg_count` 四處確認

| 用途 | 位置 | 程式碼 | 狀態 |
|------|------|--------|------|
| 遞增 | main.cu:586,615 | `vel_avg_count++` | PASS |
| VTK 正規化 | fileIO.h:678 | `1.0/((double)vel_avg_count*(double)Uref)` | PASS |
| VTK header 寫入 | fileIO.h:886 | `vel_avg_count=` | PASS |
| VTK header 讀取 | fileIO.h:373→470 | `vel_avg_count_from_vtk → vel_avg_count` | PASS |

### `rey_avg_count` 四處確認

| 用途 | 位置 | 程式碼 | 狀態 |
|------|------|--------|------|
| 遞增 | main.cu:578,608 | `rey_avg_count++` | PASS |
| RS 正規化 | fileIO.h:717 | `1.0/(double)rey_avg_count` | PASS |
| accu.dat 寫入 | fileIO.h:234 | `fp_accu << rey_avg_count` | PASS |
| accu.dat 讀取 | fileIO.h:277 | `fp_accu >> rey_avg_count` | PASS |

### 無殘留除法引用

`accu_num` (main.cu:103) 只在 for-loop 遞增 (L569, L601)，**從未**用於統計量除法 → PASS

---

## 3. FTT 門檻邏輯 — PASS

### Stage 1 (FTT ≥ 20) — 速度平均

- Sub-step 1 (main.cu:584): `if (FTT_now >= FTT_STAGE1 && step > 0)` → `Launch_AccumulateTavg()`
- Sub-step 2 (main.cu:613): `if (FTT_now >= FTT_STAGE1)` → `Launch_AccumulateTavg()`
  - `step > 0` guard 在 sub-step 2 不需要（step 已 ≥ 1）
- FTT=20 對應 step ≈ 894,726 → 低於此步數不會觸發 → PASS

### Stage 2 (FTT ≥ 50) — Reynolds stress

- Sub-step 1 (main.cu:576): `if (FTT_now >= FTT_STAGE2 && (int)TBSWITCH)` → `Launch_TurbulentSum()`
- Sub-step 2 (main.cu:606): 同上
- FTT=50 對應 step ≈ 2,236,816 → 低於此步數不會觸發 → PASS

### AccumulateUbulk 已從 streaming 移除

- evolution.h:131: `// AccumulateUbulk removed from streaming`
- 唯一呼叫在 `Launch_ModifyForcingTerm()` (evolution.h:174) → PASS

### 無殘留累積路徑

全文搜索 `Launch_TurbulentSum` 和 `Launch_AccumulateTavg`，僅在上述 FTT-gated 位置出現 → PASS

---

## 4. Checkpoint 完整性 — PASS

### Binary 寫入/讀取陣列對照

| 函數 | 位置 | 陣列數 | 內容 |
|------|------|--------|------|
| `statistics_writebin_stress` | fileIO.h:238-269 | 32 | U,V,W,P + UU..WW + PU,PV,PW + KT + DUDX2..DWDZ2 + UUU..WWW |
| `statistics_readbin_stress` | fileIO.h:281-312 | 32 | 完全相同的 32 個陣列 |

**Write 32 = Read 32** → PASS

### accu.dat 計數器

- 寫入: `rey_avg_count` (fileIO.h:234) → PASS
- 讀取: `rey_avg_count` (fileIO.h:277) → PASS

### Stage flags 續跑持久性

- `vel_avg_count > 0` → `stage1_announced = true` (main.cu:485) → PASS
- `rey_avg_count > 0` → `stage2_announced = true` (main.cu:497) → PASS

### VTK 續跑恢復流程

1. `InitFromMergedVTK` 解析 header → `vel_avg_count`, `rey_avg_count` (fileIO.h:370-378) → PASS
2. `tavg_h` 讀入（新格式 ×Uref / 舊格式 ×1.0）→ main.cu:472-476 ×count → GPU → PASS
3. `rey_avg_count > 0` → `statistics_readbin_stress` (main.cu:496) → 32 arrays → GPU → PASS

### 預存問題（非本次修改）

PP 被 MeanVars 累積 (statistics.h:157) 但未被 writebin/readbin 包含。共 33 個 GPU 累積陣列，但只 checkpoint 32 個。

---

## 5. 命名規範 — PASS

### VTK 欄位名稱 (benchmark 慣例)

| VTK SCALARS | 來源 | Code 變數 | Benchmark 含義 | 正規化 |
|-------------|------|-----------|---------------|--------|
| `U_mean` (fileIO.h:932) | `vt_global` | v_tavg ÷ count ÷ Uref | 流向平均 | ÷ Uref |
| `W_mean` (fileIO.h:945) | `wt_global` | w_tavg ÷ count ÷ Uref | 法向平均 | ÷ Uref |
| `V_mean` (fileIO.h:957) | `ut_global` | u_tavg ÷ count ÷ Uref | 展向平均 | ÷ Uref |
| `uu` (fileIO.h:973) | VV/N − (V/N)² | code VV | \<u'u'\> | ÷ Uref² |
| `ww` | WW/N − (W/N)² | code WW | \<w'w'\> | ÷ Uref² |
| `vv` | UU/N − (U/N)² | code UU | \<v'v'\> | ÷ Uref² |
| `uw` | VW/N − V·W/N² | code VW | \<u'w'\> | ÷ Uref² |
| `k` | 0.5(uu+ww+vv) | — | TKE | ÷ Uref² |

### Code → Benchmark 映射

- Code u (spanwise) → Benchmark V → `V_mean`
- Code v (streamwise) → Benchmark U → `U_mean`
- Code w (wall-normal) → Benchmark W → `W_mean`

全部正確 → PASS

### 無殘留舊名稱

`v_time_avg` 和 `w_time_avg` 僅出現在 `InitFromMergedVTK` 的 backward-compat 讀取 (fileIO.h:457,460)，不在 VTK 寫出中 → PASS

### Backward compat 讀取

- 新格式: `U_mean` → `v_tavg_h` × Uref (fileIO.h:451) → PASS
- 舊格式: `v_time_avg` → `v_tavg_h` × 1.0 (fileIO.h:457) → PASS
- 舊格式: `tavg_count=` → `vel_avg_count` (fileIO.h:381-382) → PASS

---

## 6. 統計量在定期 VTK 中不被寫入 — PASS

### 33 個 GPU 累積陣列分類

| 類別 | 陣列 | 數量 | 定期 VTK？ | 說明 |
|------|------|------|-----------|------|
| MeanVars (用於 RS) | U, V, W, UU, VV, WW, VW | 7 | 間接（正規化衍生） | 讀到暫存 buffer 計算 RS |
| MeanVars (其他) | P, PP, UV, UW, PU, PV, PW, KT | 8 | **否** | 僅 binary checkpoint |
| MeanDerivatives | DUDX2..DWDZ2 | 9 | **否** | 僅 binary checkpoint |
| 3rd-order | UUU..WWW | 9 | **否** | 僅 binary checkpoint |

### 定期 VTK 實際輸出

8 個 **正規化衍生欄位**: U_mean, W_mean, V_mean (÷ Uref), uu, ww, vv, uw, k (÷ Uref²)

- 從 7 個 GPU 陣列讀取到 **暫存 host buffer** (fileIO.h:702-715)
- 正規化計算後寫入 VTK (fileIO.h:717-737)
- 暫存 buffer 立即釋放 (fileIO.h:739-740)
- **GPU 上的 raw accumulated sums 不受影響** → PASS

### Binary checkpoint 觸發條件

`rey_avg_count > 0 && (int)TBSWITCH` (main.cu:679) → 僅 Stage 2 啟動後寫入 → PASS

---

## 總結

| # | 檢查項 | 結果 |
|---|--------|------|
| 1 | 編譯層面（宣告、簽名、include） | **PASS** |
| 2 | 計數器一致性 (vel/rey_avg_count) | **PASS** |
| 3 | FTT 門檻邏輯（無殘留累積路徑） | **PASS** |
| 4 | Checkpoint 完整性（寫入=讀取） | **PASS** |
| 5 | 命名規範（benchmark 慣例） | **PASS** |
| 6 | 統計量不在定期 VTK 中 | **PASS** |

### 預存問題備忘（非本次修改引入）

- **PP 陣列未 checkpoint**: 被 MeanVars 累積但 `statistics_writebin_stress` 未包含。需要時可補上。
- **accu_num 變數**: 仍在 for-loop 中遞增但從未使用，為無害 dead code。
