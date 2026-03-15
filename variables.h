#ifndef VARIABLES_FILE
#define VARIABLES_FILE

// ================================================================
// 1. 數學常數
// ================================================================
#define     pi      3.14159265358979323846264338327950
#define     cs      (1.0/1.732050807568877)    // 1/√3, LBM 聲速

// ================================================================
// 2. 物理域幾何
// ================================================================
#define     LX      (4.5)       // 展向 (spanwise) 長度
#define     LY      (9.0)       // 流向 (streamwise) 長度 = hill-to-hill 週期長度
#define     LZ      (3.036)     // 法向 (wall-normal) 長度
#define     H_HILL  (1.0)       // hill 高度 (Re_h 參考長度)

// ================================================================
// 3. 網格設定
// ================================================================
// 全域格點數
#define     NX      32          // 展向格點數
#define     NY      128         // 流向格點數
#define     NZ      128         // 法向格點數

// MPI 分區
#define     jp      8           // GPU 數量 (流向分割)

// 含 ghost zone 的陣列維度 (自動計算，勿手動修改)
#define     NX6     (NX+7)      // 展向: +7 (3 buffer + 1 center + 3 buffer)
#define     NYD6    (NY/jp+7)   // 流向 per-rank: NY/jp + 7
#define     NY6     (NY+7)      // 流向 global: NY + 7
#define     NZ6     (NZ+6)      // 法向: +6 (3 buffer each side)
#define     GRID_SIZE (NX6 * NYD6 * NZ6) // per-rank 總格點數

// 非均勻網格
#define     CFL 0.5    // 降低 CFL: 0.5→0.25 修正壁面插值振盪 & ω₁ 遠離 2.0 極限
#define     minSize             ((LZ-1.0)/(NZ6-6)*CFL)
#define     Uniform_In_Xdir     1   // 1=均勻, 0=非均勻
#define     Uniform_In_Ydir     1
#define     Uniform_In_Zdir     0   // 法向非均勻 (壁面加密)

// 展向映射參數
#define     LXi     (10.0)

// ================================================================
// 4. 物理參數
// ================================================================
#define     Re      700         // Reynolds number (基於 H_HILL 和 Uref)
#define     Uref    0.037      // 參考速度 (bulk velocity)
                                // Re700:0.0583, Re1400/2800:0.0776
                                // Re5600:0.0464, Re10595:0.0878
                                // 限制: Uref ≤ cs = 0.1732 (Ma < 1)
#define     niu     (Uref/Re)   // 運動黏度

// ================================================================
// 5. 時間步進
// ================================================================
#define     dt      minSize     // 直角坐標系 dt = minSize (c=1)
                                // 曲線坐標系用 dt_global (runtime 計算)

// Flow-Through Time (FTT)
// 一個 FTT = LY/Uref 個 lattice time steps
// 第 n 步的 FTT = n × dt_global × Uref / LY
#define     flow_through_time   (LY / Uref)

// ================================================================
// 6. 碰撞算子選擇 (三擇一)
// ================================================================
//  模式         USE_MRT   USE_CUMULANT   說明
//  ─────────────────────────────────────────────────────────────
//  BGK/SRT        0          0           單鬆弛 (baseline, fastest)
//  MRT            1          0           多鬆弛矩空間 (Suga 2015)
//  Cumulant-AO    0          1           累積量 All-One (Geier 2015)
//  Cumulant-WP    0          1           累積量 Well-Parameterized (Gehrke 2022)
//                                        (由 USE_WP_CUMULANT 選擇 AO/WP)
//
//  ★ USE_MRT 與 USE_CUMULANT 不可同時為 1 ★
// ================================================================
#define     USE_MRT             0
#define     USE_CUMULANT        1

// ── Cumulant 子選項 (僅 USE_CUMULANT=1 時生效) ──────────────────
//   USE_WP_CUMULANT = 0  →  AO: ω₂–ω₁₀ = 1 (Geier 2015), 全抑制, 穩定但耗散
//   USE_WP_CUMULANT = 1  →  WP: ω₃–ω₅ 從 ω₁,ω₂ 優化 (Eq.14-16),
//                             4 階平衡態 A,B (Eq.17-18),
//                             λ-limiter 正則化 (Eq.20-26)
//
//   CUM_LAMBDA: WP 正則化參數 λ
//     1e-6  → ≈AO (limiter 幾乎關閉)
//     1e-2  → Gehrke 預設 (多數情況適用)
//     1e-1  → Re≥10600 中等網格 (GR22 Table 7)
// ================================================================
#define     USE_WP_CUMULANT     0   // AO mode: all higher-order omega=1, A=B=0. Stable baseline for GILBM.
#define     CUM_LAMBDA          1.0e-2
//正則化參數引入

// ── GILBM 安全: WP 三階鬆弛率下限 ──────────────────
//   當 ω₁ > 14/9 ≈ 1.556 時，Eq.15 的 ω₄ → 0 (極點效應)
//   導致反對稱三階 cumulant 幾乎不衰減 → GILBM 插值噪聲累積 → 發散
//
//   CUM_WP_OMEGA_MIN: ω₃/ω₄/ω₅ 的最小值
//     0.5  → 至少 50% 衰減/步 (推薦: 穩定且保留部分 WP 優化)
//     0.8  → 保守 (接近 AO, 80% 衰減)
//     不定義 → 不限制 (原始 WP 行為, standard LBM 用)
//
//   目前 ω₁ = 1.80 → ω₄_raw = 0.067 → clamp 至 0.5
#define     CUM_WP_OMEGA_MIN    0.5

// ── Cumulant omega2 (bulk viscosity relaxation rate) ──────────────
//   omega2 控制體積黏度 (bulk viscosity) 的鬆弛速率
//   Gehrke 2022 Section 3.2.3: C-AO 和 C-P 兩種模式均使用 ω₂ = 1.0
//   Geier 2015: 同樣使用 ω₂ = 1.0 (AO: ω_{C≥2} = 1)
//
//   ω₂ = 1.0 → 體積黏度 trace 完全鬆弛至平衡態 (ζ_bulk = 0)
//               消除 trace 的非平衡記憶，提高穩定性
//   ω₂ = 0.5 → trace 部分鬆弛 (ζ_bulk = cs²·Δt/6)，會保留壓力波非平衡態
//               可能與 GILBM 插值交互作用導致不穩定
//
//   影響: omega3-5 (Eq.14-16) 和 A,B (Eq.17-18) 的奇異點位置
//   ★ 此全域變數同時控制 cumulant_collision.h 和 main.cu 診斷 ★
#define     CUM_OMEGA2          1.0

// ── 力的施加方案 & Galilean 校正 (DEBUG 開關) ──────────────
//   CUM_SIGNFLIP:  1 = Strang sign-flip (Gehrke §3.2.1), 0 = 不翻轉
//   CUM_GUO_SRC:   1 = Guo explicit source (方案 B), 0 = 不加 source
//   CUM_GALILEAN:  1 = Galilean correction (Eq.3.70-3.75), 0 = 不修正
//
//   方案 A (Gehrke thesis):  SIGNFLIP=1, GUO_SRC=0, GALILEAN=1  (標準 LBM 用)
//   方案 B (Guo explicit):   SIGNFLIP=0, GUO_SRC=1, GALILEAN=0  (GILBM 推薦)
//   純 debug (無力):          SIGNFLIP=0, GUO_SRC=0, GALILEAN=0
//
//   ★ GILBM 必須用方案 B ★
//   方案 A 的 Strang sign-flip 假設 exact lattice-shift streaming，
//   但 GILBM 用 7-point Lagrange interpolation streaming，
//   sign-flip + Galilean correction 的 u² 項會放大插值噪聲 → 指數發散。
//
//   方案 B 的 Guo explicit source 不依賴 exact advection，
//   且 (1-ω/2) prefactor 正確處理碰撞-遷移的時間偏移。
#define     CUM_SIGNFLIP        0   // GILBM: 關閉 Strang sign-flip
#define     CUM_GUO_SRC         1   // GILBM: 啟用 Guo explicit source
#define     CUM_GALILEAN        0   // GILBM: 關閉 Galilean correction (u² 噪聲放大)

// ── Pre-collision Regularization (穩定化 GILBM+Cumulant 的關鍵) ──
//   Chimera 變換的基底依賴局部速度 u。GILBM 的 7-point Lagrange 插值
//   在分佈函數中引入高階噪聲（3 階及以上），這些噪聲通過 Chimera 的
//   速度依賴基底變換被放大 → 正反饋 → 指數發散。
//
//   正則化在碰撞前將 f 投影到物理子空間：
//     f_reg = f^eq(ρ,u) + f^neq_2nd(Π^neq)
//   保留: 密度(0階)、動量(1階)、應力張量(2階)
//   移除: 3階及以上的插值噪聲
//
//   與 AO 模式完全相容（AO 碰撞也將 3+ 階歸零）
//   理論依據: Latt & Chopard 2006, Malaspinas 2015
//
//   1 = 啟用正則化 (推薦: GILBM 環境)
//   0 = 不正則化 (僅標準 LBM 用)
#define     CUM_REGULARIZE      1

// ── 插值方案選擇 ──────────────────────────────────────────────
//   GILBM_INTERP_IDW = 0 → 7-point Lagrange (原始方案, 6 階精度但有 Runge 振盪風險)
//   GILBM_INTERP_IDW = 1 → 7-point IDW (Inverse Distance Weighting, 單調性保證)
//
//   IDW 性質:
//     - 權重 w_k = 1/|t-k|^p，全部 ≥ 0 且 Σw=1
//     - 結果必在輸入值的 [min, max] 之內 → 不產生新極值 (monotone)
//     - 消除 Runge phenomenon（7 點 Lagrange 在 stencil 邊緣的振盪）
//     - 代價: 精度從 O(h^6) 降至 O(h^2)，對穩態求解可接受
//
//   IDW_POWER: 距離權重的冪次
//     p=2 → 標準 IDW，適度平滑
//     p=3 → 更集中於近鄰點，減少遠距離影響
//     p=4 → 近似最近鄰，非常局部
#define     GILBM_INTERP_IDW    1       // 1=IDW, 0=Lagrange
#define     GILBM_IDW_POWER     2.0     // IDW 距離權重冪次

// ── 診斷: 純平衡態壁面 BC (關閉 CE 非平衡修正) ──
//   設為 1 時，ChapmanEnskogBC 只返回 f_eq (C_alpha=0)，
//   用於測試壁面非平衡修正是否為不穩定性來源。
//   正式計算請設為 0。
#define     WALL_EQ_ONLY        0

// ── Odd-Even Filter (已移除: 方案 C 因降低有效 Re 而放棄) ──
// 改用方案 A: Matrix-based moment transform 取代 Chimera
// #define     CUM_ODDEVEN_SIGMA   0.05

// ── 互斥檢查 ──
#if USE_MRT && USE_CUMULANT
#error "USE_MRT and USE_CUMULANT are mutually exclusive. Set only one to 1."
#endif

// ================================================================
// 7. Kernel 策略
// ================================================================
// 1 = P6 (Step1 → MPI → Step23), 推薦
// 0 = Old (pre-copy → Buffer → Full → MPI → Correction), 用於比較測試
// P5 (combined MRT) 在 USE_MRT=1 時自動啟用
#define     USE_P6              1

// ================================================================
// 8. 模擬控制
// ================================================================
// loop 已移除：模擬以 FTT_STOP 為唯一終止條件
#define     NDTMIT      50      // 每 N 步輸出 monitor 資料
#define     NDTFRC      20     // 每 N 步更新外力項 (CFL=0.25 下更頻繁檢查)
#define     NDTBIN      100000   // 每 N 步輸出 binary checkpoint
#define     NDTVTK      1000    // 每 N 步輸出 VTK

// ====================================================================
// Hybrid Dual-Stage Force Controller (PID + Gehrke multiplicative)
// ====================================================================
// Phase 1 (PID):    |Re%| > SWITCH_THRESHOLD — 冷啟動/遠離目標安全加速
//   Force = Kp*error*norm + integral + Kd*d_error*norm
//   norm = Uref²/LY
// Phase 2 (Gehrke): |Re%| ≤ SWITCH_THRESHOLD — 穩態乘法微調
//   Gehrke ref: Gehrke & Rung (2020) Int J Numer Meth Fluids, Sec 3.1
//   原文: F *= (1 - 0.1 × Re%)  當 |Re%| > 1.5%, 每 FTT 更新 10 次
//   Correction clamp: multiplier ∈ [0.5, 1.5] (prevents single-step catastrophe)
// 連續 Mach brake 在兩模式之上統一適用
// ====================================================================

// PID controller gains (Phase 1)
#define     FORCE_KP                2.0     // 比例增益
#define     FORCE_KI                0.3     // 積分增益
#define     FORCE_KD                0.5     // 微分增益

// Gehrke multiplicative controller (Phase 2)
// ★ gain 必須配合更新頻率! 論文: gain=0.1 @ 10 updates/FTT
//   我們 NDTFRC=20 → ~12000 updates/FTT → gain_eff = 0.1 × 10/12000 ≈ 8e-5
//   或者直接降低 SWITCH_THRESHOLD 限制 Gehrke 只在小 Re% 工作
#define     FORCE_GEHRKE_GAIN       0.1     // 論文原值: 0.1 (F *= 1 - gain × Re%)
#define     FORCE_GEHRKE_DEADZONE   1.5     // 論文原值: 1.5% (|Re%| < 1.5% → hold)
#define     FORCE_GEHRKE_FLOOR      0.1     // Force 下限 = 10% × F_Poiseuille (防 Force→0 陷阱)

// Controller switching
#define     FORCE_SWITCH_THRESHOLD  5.0     // |Re%| ≤ 5% → Gehrke (論文適用範圍)
                                             // correction 極值 = 1 ± 0.1×5 = [0.5, 1.5]
                                             // ★ 10% 時 correction=1.9 → 每步翻倍 → 發散!

// Force magnitude cap (防止任何模式下 Force 失控)
#define     FORCE_CAP_MULT          50.0    // Force 上限 = 50 × F_Poiseuille

// Cold start ramp: 已關閉 (D3Q19 Edit3 無此功能，對齊後停用)
// 設 RAMP_STEPS=0 即完全跳過 ramp 邏輯
#define     FORCE_RAMP_STEPS        0       // 0=關閉 (對齊 D3Q19 Edit3)
#define     FORCE_RAMP_CAP          15.0    // (ramp 關閉時此值無效)

// Mach safety brake (continuous, both phases)
#define     MA_BRAKE_MULT_THRESHOLD 1.7     // Ma_max > 1.7×Ma_bulk → 開始二次衰減
#define     MA_BRAKE_MULT_CRITICAL  2.1     // Ma_max > 2.1×Ma_bulk → 緊急歸零
#define     MA_BRAKE_GROWTH_LIMIT   0.30    // Ma_max 單步增長 >30% → 額外 ×0.3

// ================================================================
// 9. FTT 閾值與統計控制
// ================================================================
// Stage 0: FTT < FTT_STATS_START  → 只跑瞬時場，不累積統計量
// Stage 1: FTT ≥ FTT_STATS_START  → 所有 33 個統計量同時累積 (共用 accu_count)
#define     FTT_STATS_START     20.0    // 統計量開始累積
#define     FTT_STOP            100.0   // 模擬結束

// VTK 輸出等級
// 0 = 基本 (13 SCALARS): 瞬時速度(3) + 瞬時渦度(3) + 平均速度(2) + RS(3) + k_TKE + P_mean
// 1 = 完整: Level 0 + V_mean + 展向RS(uv,vv,vw) + 平均渦度 + ε + Tturb + Pdiff + PP_RS
#define     VTK_OUTPUT_LEVEL    0

// ================================================================
// 10. 重啟 (Restart) 配置
// ================================================================
// INIT: 重啟模式
//   0 = 冷啟動 (zero velocity, ρ=1, 2×Poiseuille Force)
//   1 = 從 per-rank binary 續跑 (legacy, 只有瞬時場)
//   2 = 從 merged VTK 續跑 (f=feq 近似, 統計量無法還原)
//   3 = 從 binary checkpoint 續跑 (精確: f + 統計量累積和)
#define     INIT                (0)     // Cold start for Cumulant test (switch back to 3 after verified stable)

// INIT=2 用: merged VTK 檔案路徑
#define     RESTART_VTK_FILE    "result/velocity_merged_31001.vtk"

// INIT=3 用: binary checkpoint 目錄路徑
#define     RESTART_BIN_DIR     "checkpoint/step_70001"

// 統計量讀取 (僅 INIT=1 時生效)
// 1 = 從 statistics/*.bin 讀取上次累積的統計量 + accu.dat
// 0 = 不讀取，FTT ≥ FTT_STATS_START 後從零累積
#define     TBINIT              (0)

// ================================================================
// 11. 初始擾動 (觸發 3D 湍流轉捩)
// ================================================================
// 湍流建立後設為 0
#define     PERTURB_INIT        0       // 1=注入隨機擾動, 0=不擾動
#define     PERTURB_PERCENT     5       // 擾動振幅 (% of Uref), 典型 1-10%

// ================================================================
// 12. GPU 設定
// ================================================================
#define     NT      32          // CUDA block size (x 方向 thread 數)

// ================================================================
// 待移除 (deprecated, 下次清理時刪除)
// ================================================================
#define     TBSWITCH    (1)     // → 已被 FTT_STATS_START 取代
                                // 目前部分舊 code 仍依賴此 flag
                                // 全部改完後刪除

#endif

/*
備註: 座標系對應關係

  Code 方向    物理方向        Benchmark 符號
  ──────────────────────────────────────────
  x (i)       展向 spanwise    V (benchmark)
  y (j)       流向 streamwise  U (benchmark)
  z (k)       法向 wall-normal W (benchmark)

  VTK 輸出時需做映射:
    VTK U_mean ← code sum_v / N / Uref
    VTK V_mean ← code sum_u / N / Uref
    VTK W_mean ← code sum_w / N / Uref

鬆弛時間:
  直角坐標系: τ = 3ν/dt + 0.5          (dt = minSize)
  曲線坐標系: ω_global = 3ν/dt_global + 0.5
  [GTS] 全域時間步: ω_global 處處相同, 無局部時間步
*/