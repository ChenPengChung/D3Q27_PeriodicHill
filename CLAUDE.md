# GILBM Project Memory & Plan

> 此檔案會被 Claude Code 自動讀取作為專案上下文
> 此專案為 CUDA+MPI 的 D3Q19 Lattice Boltzmann 週期性山丘流場模擬

---

## 一、核心物理設定

### c=1 約定（已驗證）
- **c 是固定的模型參數**，不是 c = Δx/Δt
- Imamura Eq. 2: `c_i = c × e_i`
- Imamura Eq. 13: `c̃ = c × e × ∂ξ/∂x`
- **我們的程式碼使用 c=1**：所有 c 因子消失，位移 = `dt × e × ∂ξ/∂x`
- **不需要額外乘 minSize**：minSize 隱藏在度量項中 (dk_dz = 2/minSize at wall)
- 黏度公式確認 c=1：`ν = (τ-0.5)/3 × dt`

### dt 與 tau 關係
```c
// variables.h: dt, tau 是 runtime 變數 (Phase 3 Imamura CFL time step)
extern double dt;   // 全域時間步長
extern double tau;  // 鬆弛時間
#define niu ((tau-0.5)/3.0*dt)
#define Uref (Re*niu)
```

### 網格參數 (variables.h)
```c
#define LX 4.5      // 流向長度
#define LY 9.0      // 展向長度
#define LZ 3.036    // 法向長度
#define NX 32       // 流向格數
#define NY 128      // 展向格數 (4 GPU 分割: jp=4)
#define NZ 64       // 法向格數
#define jp 4        // MPI rank 數 = GPU 數
#define NX6 (NX+7)  // 含 ghost = 39
#define NYD6 (NY/jp+7) // 每 rank 含 ghost = 39
#define NZ6 (NZ+6)  // 含 ghost = 70
#define CFL 0.6
#define minSize ((LZ-1.0)/(NZ6-6)*CFL)
#define NT 32       // CUDA block size x
```

---

## 二、專案結構

- **分支**: `Edit3_GILBM`
- **編譯環境**: CUDA + MPI, 遠端 `chenpengchung@140.114.58.87`

### 關鍵檔案清單

| 檔案 | 行數 | 說明 |
|------|------|------|
| `main.cu` | 399 | 主程式：初始化→預計算→時間迴圈→輸出 |
| `variables.h` | 109 | 所有 #define 常數、extern dt/tau |
| `common.h` | 95 | CHECK_CUDA/CHECK_MPI 巨集 |
| `memory.h` | 202 | GPU 記憶體配置/釋放 |
| `evolution.h` | 191 | kernel 調度 + Launch_CollisionStreaming() |
| `gilbm/evolution_gilbm.h` | 553 | **完整 4-step kernel** (已完成) |
| `gilbm/interpolation_gilbm.h` | 55 | Intrpl7, lagrange_7point_coeffs, compute_feq_alpha |
| `gilbm/boundary_conditions.h` | 100 | NeedsBoundaryCondition, ChapmanEnskogBC |
| `gilbm/precompute.h` | 513 | δη/δξ/δζ 預計算 |
| `gilbm/metric_terms.h` | 412 | dk_dz, dk_dy 度量項計算 |
| `gilbm/diagnostic_gilbm.h` | 512 | Phase 1.5 診斷測試 |
| `Claude_GILBM.md` | 399 | 完整理論推導文件 |

### Include 順序 (main.cu)
```
variables.h → common.h → model.h → memory.h → initialization.h
→ gilbm/metric_terms.h → gilbm/precompute.h → gilbm/diagnostic_gilbm.h
→ communication.h → monitor.h → statistics.h → evolution.h → fileIO.h
```

### evolution_gilbm.h 內部 include 順序
```
__constant__ GILBM_e, GILBM_W, GILBM_dt, GILBM_delta_eta, GILBM_delta_xi
→ gilbm/interpolation_gilbm.h  (依賴 GILBM_e, GILBM_W)
→ gilbm/boundary_conditions.h  (依賴 GILBM_e, GILBM_W)
→ 自身函數定義
```

---

## 三、記憶體索引約定（最重要！）

### 3D 索引 → 1D 線性化
```cpp
const int nface = NX6 * NZ6;             // 一個 j-slice 的大小
const int index = j * nface + k * NX6 + i;  // 全域 3D → 1D
const int idx_jk = j * NZ6 + k;            // j-k 平面 (度量項/LTS 用)
```

### f_pc_d 索引 (private post-collision stencil)
```cpp
// 大小: 19 × 343 × GRID_SIZE (GRID_SIZE = NX6*NYD6*NZ6)
// 三層索引: q(方向) × flat(stencil位置) × index(格點)
f_pc[(q * STENCIL_VOL + flat) * GRID_SIZE + index]
// 其中 flat = si*49 + sj*7 + sk, si/sj/sk ∈ [0,6]
```

### feq_d 索引
```cpp
// 大小: 19 × GRID_SIZE
feq_d[q * GRID_SIZE + index]
```

### omegadt_local_d 索引
```cpp
// 大小: GRID_SIZE (每格點一個值，雖然只跟 j,k 相關但展開成全場)
omegadt_local_d[index]  // = omega_local * dt_local (由 Init_OmegaDt_Kernel 計算)
```

### delta_zeta_d 索引
```cpp
// 大小: 19 × NYD6 × NZ6 (每方向每 j-k 點)
delta_zeta_d[q * NYD6 * NZ6 + idx_jk]
```

### f_new (streaming 後) = fd[] 或 ft[]
```cpp
// 每方向獨立陣列: f_new_ptrs[q][index]
// 由 fd[0..18] / ft[0..18] double-buffer 切換
```

---

## 四、GILBM 4-Step 演算法（Imamura 2005）

### 資料結構

| 陣列 | 大小 | 說明 |
|------|------|------|
| `f_pc_d[19×343×grid]` | ~5.55 GB | 每格點私有 post-collision |
| `feq_d[19×grid]` | ~16 MB | 全域平衡分佈 |
| `omegadt_local_d[grid]` | ~1.6 MB | ω·Δt = omega_local × dt_local |
| `dt_local_d[NYD6×NZ6]` | ~21 KB | 局部時間步長 |
| `omega_local_d[NYD6×NZ6]` | ~21 KB | 局部鬆弛頻率 ω = 1/τ_local |
| `dk_dz_d[NYD6×NZ6]` | ~21 KB | ∂ζ/∂z 度量項 |
| `dk_dy_d[NYD6×NZ6]` | ~21 KB | ∂ζ/∂y 度量項 |
| `delta_zeta_d[19×NYD6×NZ6]` | ~408 KB | ζ 方向 RK2 位移 |
| `GILBM_delta_eta[19]` (__constant__) | 152 B | η 位移 (常數) |
| `GILBM_delta_xi[19]` (__constant__) | 152 B | ξ 位移 (常數) |

### 演算法流程

```
Step 1:   從自己的 f_pc 插值 → f_new (f_streamed)
Step 1.5: 從 f_new 計算宏觀量 → feq_d, rho_out, u/v/w_out
Step 2:   重估 (Eq.35): 讀取鄰點 f_new + feq → f_re
Step 3:   碰撞 → 寫回 f_pc
```

### Step 1 詳細邏輯 (插值 + Streaming)
```cpp
for q = 0..18:
    if q==0:
        f_streamed = f_pc[(0*343 + center_flat)*GRID_SIZE + index]
    elif NeedsBoundaryCondition(q, dk_dy, dk_dz, is_bottom/top):
        f_streamed = ChapmanEnskogBC(q, rho_wall, du_dk, dv_dk, dw_dk,
                                      dk_dy, dk_dz, omega_A, dt_A)
    else:
        // 載入 7³=343 個 f_pc 值 → f_stencil[7][7][7]
        // 計算 departure point: up_i = i - a_local*delta_eta[q], ...
        // a_local = dt_A / GILBM_dt (LTS 加速因子)
        // Lagrange 權重: t_i = up_i - bi → lagrange_7point_coeffs()
        // 三步張量積: eta→xi→zeta 各做 Intrpl7
    f_new_ptrs[q][index] = f_streamed
    rho += f_streamed; mx += e[q][0]*f_streamed; ...
```

### Step 1.5 詳細邏輯 (宏觀量 + feq)
```cpp
rho_stream += rho_modify[0];          // 全域質量修正
f_new_ptrs[0][index] += rho_modify[0]; // 修正 f0
rho_A = rho_stream;
u_A = mx_stream / rho_A;  // 物理直角坐標速度 (不需 Jacobian)
v_A = my_stream / rho_A;
w_A = mz_stream / rho_A;
for q = 0..18:
    feq_d[q*GRID_SIZE + index] = compute_feq_alpha(q, rho_A, u_A, v_A, w_A);
rho_out[index] = rho_A; u_out[index] = u_A; ...
```

### Step 2+3 詳細邏輯 (重估 + 碰撞, 合併迴圈)
```cpp
for q = 0..18:
    if (need_bc) continue;  // BC 方向跳過
    for si,sj,sk = 0..6:    // 遍歷 343 個 stencil 點 B
        gi=bi+si; gj=bj+sj; gk=bk+sk;
        idx_B = gj*nface + gk*NX6 + gi;
        flat  = si*49 + sj*7 + sk;

        // Step 2: 讀取 f_B, feq_B (Gauss-Seidel: 從 f_new 讀)
        f_B = f_new_ptrs[q][idx_B];
        feq_B = feq_d[q*GRID_SIZE + idx_B]; // ghost zone 用 on-the-fly 計算
        omegadt_B = omegadt_local_d[idx_B];
        R_AB = omegadt_A / omegadt_B;
        f_re = feq_B + (f_B - feq_B) * R_AB;  // Eq.35

        // Step 3: BGK 碰撞
        f_re -= (1.0/omega_A) * (f_re - feq_B);  // Eq.3

        // 寫回 A 的私有 f_pc
        f_pc[(q*343 + flat)*GRID_SIZE + index] = f_re;
```

---

## 五、關鍵函數 API

### interpolation_gilbm.h
```cpp
// 7 點加權求和巨集
#define Intrpl7(f1,a1, f2,a2, f3,a3, f4,a4, f5,a5, f6,a6, f7,a7)

// 計算 1D 7 點 Lagrange 插值係數，節點在整數位置 0..6，求值位置 t
__device__ void lagrange_7point_coeffs(double t, double a[7]);

// D3Q19 平衡分佈函數 (物理直角坐標，不需 Jacobian)
// feq = w_α · ρ · (1 + 3·(e·u) + 4.5·(e·u)² − 1.5·|u|²)
__device__ double compute_feq_alpha(int alpha, double rho, double u, double v, double w);
```

### boundary_conditions.h
```cpp
// 判定方向 α 是否需要壁面 BC（ẽ^ζ_α 指向壁外）
// 底壁: ẽ^ζ > 0 → true;  頂壁: ẽ^ζ < 0 → true
// ẽ^ζ_α = e_y[α]·dk_dy + e_z[α]·dk_dz
__device__ bool NeedsBoundaryCondition(int alpha, double dk_dy, double dk_dz, bool is_bottom);

// Chapman-Enskog BC (Imamura Eq. A.9, 6 項張量展開, c=1 約定)
// f = w_α · ρ_wall · (1 + C_α)
// C_α = -ω·Δt × Σ [3·c_iα·c_iβ - δ_αβ] · (du_α/dk)·(dk/dx_β)
// 注意: c=1 時係數為 3 (= 1/c_s^4 = 1/(1/3)^2 = 9 → ×c² = 9×(1/3) = 3)
// 輸入: du_dk, dv_dk, dw_dk (二階單邊差分), dk_dy, dk_dz, omega=omega_A, localtimestep=dt_A
__device__ double ChapmanEnskogBC(int alpha, double rho_wall,
    double du_dk, double dv_dk, double dw_dk,
    double dk_dy, double dk_dz, double omega, double localtimestep);
```

### evolution_gilbm.h 核心函數
```cpp
// 計算 stencil 起始點，含邊界 clamping
// k 方向: bk ∈ [3, NZ6-10] (確保 bk+6 = NZ6-4，stencil 不超出壁面)
__device__ void compute_stencil_base(int i, int j, int k, int &bi, int &bj, int &bk);

// 讀取 19 個 f_new 計算宏觀量
__device__ void compute_macroscopic_at(double *f_ptrs[19], int idx,
    double &rho, double &u, double &v, double &w);

// 核心 4-step 函數 (Buffer 和 Full kernel 共用)
__device__ void gilbm_compute_point(int i, int j, int k,
    double *f_new_ptrs[19], double *f_pc, double *feq_d, double *omegadt_local_d,
    double *dk_dz_d, double *dk_dy_d, double *delta_zeta_d,
    double *dt_local_d, double *omega_local_d,
    double *u_out, double *v_out, double *w_out, double *rho_out,
    double *Force, double *rho_modify);
```

### Kernel 列表 (evolution_gilbm.h)
```cpp
// 內部區域 kernel (全 j 範圍)
__global__ void GILBM_StreamCollide_Kernel(...);
// guard: if (i<=2 || i>=NX6-3 || k<=2 || k>=NZ6-3) return;

// 緩衝區 kernel (指定 start j 偏移)
__global__ void GILBM_StreamCollide_Buffer_Kernel(..., int start);

// 初始化 kernels
__global__ void Init_FPC_Kernel(...);    // f[q][idx_B] → f_pc[...]
__global__ void Init_Feq_Kernel(...);    // f → rho,u → feq_d
__global__ void Init_OmegaDt_Kernel(...);// omega_dt = tau_local * dt_local
```

---

## 六、Kernel 調度流程 (evolution.h)

```
Launch_CollisionStreaming(f_old[19], f_new[19]):
  1. cudaMemcpy(f_new ← f_old)  // Double-buffer 預拷貝
  2. Buffer kernel (j=3, stream1)     // 緩衝行先算 (stream overlap)
  3. Buffer kernel (j=NYD6-7, stream1)
  4. AccumulateUbulk (stream1)
  5. Full kernel (全 j, stream0)
  6. MPI ISend/IRecv y 方向邊界交換
  7. MPI_Waitall
  8. periodicSW (x 方向週期邊界)
```

### 時間迴圈 (main.cu)
```
for step = 0..loop:
    Launch_CollisionStreaming(ft, fd)  // 偶數步
    Launch_CollisionStreaming(fd, ft)  // 奇數步 (step += 1)
    每 NDTFRC 步: Launch_ModifyForcingTerm()
    每 NDTMIT 步: Launch_Monitor()
    每 1000 步: VTK 輸出
    每步: 全域質量守恆修正
```

---

## 七、當前進度

### evolution_gilbm.h 狀態 — 全部完成

| 部分 | 狀態 | 備註 |
|------|------|------|
| `__constant__` 宣告 | ✅ | GILBM_e, GILBM_W, GILBM_dt, delta_eta/xi |
| `compute_stencil_base()` | ✅ | 含邊界 clamping (k: [2, NZ6-9]) |
| `compute_macroscopic_at()` | ✅ | D3Q19 宏觀量計算 |
| **Step 1: 插值 + Streaming** | ✅ | 7pt Lagrange 三步張量積, wall BC |
| **Step 1.5: 宏觀量 + feq** | ✅ | 質量修正 + feq_d + rho/u/v/w_out |
| **Step 2: 重估 (Eq.35)** | ✅ | R_AB = omegadt_A/omegadt_B, ghost zone on-the-fly feq |
| **Step 3: 碰撞 (Eq.3)** | ✅ | BGK with 1/omega_A → f_pc |
| **Kernel 包裝函數** | ✅ | Full/Buffer/Init_FPC/Init_Feq/Init_OmegaDt |

### 模擬狀態
- 所有診斷測試通過 (Phase 0, 1.5, 2, 3, 4)
- 已可完整運行，VTK 輸出正常
- Force driving 已啟用，Ub_avg 從 0 開始增長

### 已完成的驗證 (Audit)
- **feq 審計**: 確認 `compute_feq_alpha` 在曲線坐標 GILBM 中不需 Jacobian 修正
- **CE BC 張量**: c=1 約定下係數為 `3.0` (非 `9.0`)
- **位移公式**: δη/δξ 以 dt_global 預計算，kernel 中乘 a_local 縮放至 dt_local
- **Departure point clamp**: CFL < 1 保證 |位移| < 1 格，clamp 為安全網不會觸發

---

## 八、關鍵公式

### LTS 重估 (R_AB)
```
R_AB = (ω_A·Δt_A) / (ω_B·Δt_B) = omegadt_A / omegadt_B
```
uniform grid 時 R_AB = 1 → f̃ = f（退化為標準 LBM）

### 壁面 BC (Chapman-Enskog, Eq. A.9, c=1)
```
f_α = w_α · ρ_wall · (1 + C_α)
C_α = -ω·Δt × { 6 項 tensor 展開 }
    = -omega*dt × Σ_α,β [3·c_iα·c_iβ - δ_αβ] · (du_α/dk)·(dk/dx_β)
```
- 係數 3 = 1/c_s^2 (c=1 時 c_s²=1/3 → 1/c_s^2=3)；c≠1 時為 9·c_iα·c_iβ - 3·δ_αβ
- `du/dk|wall = (4·u[k±1] - u[k±2]) / 2` (二階單邊差分, u[wall]=0)
- `rho_wall = rho[k=4]` (零法向壓力梯度近似, 壁面 k=3)
- dk/dx = 0 → 只有 β=y,z 存活 → 3α × 2β = 6 項

### 位移公式
```
δη = dt_global × e_x / dx        (均勻 x, __constant__ GILBM_delta_eta[19])
δξ = dt_global × e_y / dy        (均勻 y, __constant__ GILBM_delta_xi[19])
δζ = dt_local × ẽ^ζ(k_half)      (非均勻 z, RK2 中點, delta_zeta_d[19×NYD6×NZ6])
```
- δη/δξ 以 `dt_global` 預計算，存在 __constant__ memory
- Kernel 中乘 LTS 加速因子: `delta_eta_loc = a_local * GILBM_delta_eta[q]`
- `a_local = dt_A / GILBM_dt` (dt_A = dt_local, GILBM_dt = dt_global)
- δζ 已用 `dt_local` 預計算 (precompute.h)，kernel 中直接使用不需縮放

---

## 九、網格結構

### z (k) 方向 — 非均勻 + 壁面 (buffer=3)
```
k=0,1        ghost layer (線性外插)
k=2          buffer layer (外插: 2*z[3]-z[4])
k=3          底壁計算點 (C-E BC, z=Hill(y))
k=4..NZ6-5   內部計算點
k=NZ6-4      頂壁計算點 (C-E BC, z=LZ)
k=NZ6-3      buffer layer (外插: 2*z[NZ6-4]-z[NZ6-5])
k=NZ6-2,NZ6-1  ghost layer (線性外插)
```

### x (i) 方向 — 均勻 + 週期
```
i=0,1,2       ghost/buffer (periodicSW 填充)
i=3..NX6-4    計算區域
i=NX6-3,NX6-2,NX6-1  ghost/buffer
```

### y (j) 方向 — 均勻 + MPI 分割
```
j=0,1,2       ghost/buffer (MPI halo)
j=3..NYD6-4   計算區域
j=NYD6-3,NYD6-2,NYD6-1  ghost/buffer
```

### Kernel Guards
- Full kernel: `if (i<=2 || i>=NX6-3 || k<=2 || k>=NZ6-3) return;`
- Init kernel: `if (i>=NX6 || j>=NYD6 || k>=NZ6) return;`

---

## 十、座標映射

```
物理座標 (x,y,z) → 計算空間 (η=i, ξ=j, ζ=k)
| ∂η/∂x  ∂η/∂y  ∂η/∂z |   | 1/dx   0      0      |
| ∂ξ/∂x  ∂ξ/∂y  ∂ξ/∂z | = | 0      1/dy   0      |  ← 常數
| ∂ζ/∂x  ∂ζ/∂y  ∂ζ/∂z |   | 0      dk_dy  dk_dz  |  ← 空間變化
```
- dk_dy, dk_dz 由 `ComputeMetricTerms()` 計算 (metric_terms.h)
- 只有 ζ 行含空間變化項 → 只有 z 方向需要 RK2 位移預計算

---

## 十一、編譯與部署

### 遠端伺服器
- `chenpengchung@140.114.58.87:/home/chenpengchung/D3Q27_PeriodicHill`
- 同步工具: `.vscode/Zsh_mainsystem.sh` (rsync-based)
- 指令: `mobaxterm autopush 87` / `mobaxterm autopull 87`

### 編譯
```bash
nvcc -O2 -arch=sm_80 main.cu -lmpi -o main.exe
# 或用 tasks.json 中的 [Mac] Compile + Run
```

---

## 十二、變數命名對照 (Imamura ↔ 程式碼)

| Imamura 符號 | 程式碼變數 | 說明 |
|-------------|-----------|------|
| ω (鬆弛頻率) | `omega_A`, `omega_local_d` | = 1/τ_local |
| ω·Δt (教科書 τ) | `omegadt_A`, `omegadt_local_d` | = omega_local × dt_local |
| Δt_local | `dt_A`, `dt_local_d` | 局部時間步長 |
| τ_local | (非直接儲存) | = 0.5 + 1/(3·Re·dt_local) |
| a (LTS 加速因子) | `a_local` | = dt_A / GILBM_dt |

---

## 十三、參考文件

- Imamura 2005, J. Comput. Phys. 202, 645-663
- `Claude_GILBM.md` — 完整理論推導 (399 行)
- `~/.claude/plans/bright-wandering-starfish.md` — 詳細 Plan

### 時間步長與碰撞算子的層級關係

#### 1. 直角坐標系：全域時間步長
- `dt = minSize`（由 CFL 條件決定）
- 全域鬆弛時間 `tau = 0.6833`
- 全域黏度：`ν = (tau - 0.5)/3 × dt`

#### 2. 曲線坐標系：全域碰撞算子
- Lattice Boltzmann Equation 在曲線座標 (η, ξ, ζ) 中操作
- 碰撞前綴為 `1/tau`，其中 `tau` 由全域定義（= 0.6833）
- 此時 `dt_global / tau` 控制碰撞步的物理時間推進

#### 3. 曲線坐標系 + Local Time Stepping (LTS)：局部碰撞算子
- 每個格點的 `dt_local` 不同（由局部 CFL 決定）
- 碰撞前綴改用 **局部鬆弛頻率** `ω_local`：
```
ω_local = 1 / tau_local
tau_local = 0.5 + 1/(3 × Re × dt_local)
```
- 等價形式：
```
ω_local = (3 × Re × dt_local) / (1 + 1.5 × Re × dt_local)
```
- **物理意義**：`dt_local` 越小（壁面附近），`tau_local` 越大，碰撞越弱；
    `dt_local` 越大（遠離壁面），`tau_local → 0.5 + ε`，趨近穩定性極限
- LTS 重估比值：`R_AB = (ω_A·Δt_A) / (ω_B·Δt_B) = omegadt_A / omegadt_B`

> **一致性檢查**：當 `dt_local = dt_global = minSize` 時，
> `tau_local = 0.5 + 1/(3·Re·minSize) = tau`，退化為全域碰撞算子

