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

// 非均勻網格
#define     CFL                 0.5
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
#define     Uref    0.0583      // 參考速度 (bulk velocity)
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
// 6. 碰撞算子
// ================================================================
// 0 = BGK/SRT (Single Relaxation Time)
// 1 = MRT (Multi-Relaxation-Time)
#define     USE_MRT             1

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
#define     loop        500000  // 最大時間步數
#define     NDTMIT      50      // 每 N 步輸出 monitor 資料
#define     NDTFRC      100     // 每 N 步更新外力項
#define     NDTBIN      10000   // 每 N 步輸出 binary checkpoint
#define     NDTVTK      1000    // 每 N 步輸出 VTK

// 外力控制器增益 (P controller, Phase 1: additive)
// Re=100: alpha=10, Re=2800: alpha=3~14
// 週期山丘需較高 gain 加速收斂
#define     force_alpha 3

// Gehrke & Rung (2022) 雙階段外力控制器
// Phase 1 (P-additive): |Re%| > THRESHOLD → 原始 P 控制器 (冷啟動/遠離目標)
// Phase 2 (Gehrke-mult): |Re%| ≤ THRESHOLD → 乘法微調 (接近目標)
#define     FORCE_SWITCH_THRESHOLD  8.0    // Re% 切換門檻 (%)

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
#define     INIT                (3)

// INIT=2 用: merged VTK 檔案路徑
#define     RESTART_VTK_FILE    "result/velocity_merged_1800001.vtk"

// INIT=3 用: binary checkpoint 目錄路徑
#define     RESTART_BIN_DIR     "checkpoint/step_1800001"

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
  局部時間步: ω_local(j,k) = 3ν/dt_local(j,k) + 0.5
*/