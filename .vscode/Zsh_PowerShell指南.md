# D3Q27 PeriodicHill — 完整操作手冊

> **整合自**：SHORTCUTS.md + MAC_VSCODE_SETUP.md + MAC_TEST_REPORT.md  
> **適用系統**：Windows (PowerShell + PuTTY) / macOS (Bash + rsync + sshpass)  
> **最後更新**：2026-02-14

---

## 目錄

1. [環境需求與首次設定](#1-環境需求與首次設定)
2. [MobaXterm 同步命令總覽 (Git-like)](#2-mobaxterm-同步命令總覽-git-like)
3. [Push 系列 — 上傳到遠端](#3-push-系列--上傳到遠端)
4. [Pull 系列 — 從遠端下載](#4-pull-系列--從遠端下載)
5. [Fetch 系列 — 完整同步（含刪除）](#5-fetch-系列--完整同步含刪除)
6. [狀態檢查與比對](#6-狀態檢查與比對)
7. [背景自動上傳 (watchpush)](#7-背景自動上傳-watchpush)
8. [背景自動下載 (watchpull)](#8-背景自動下載-watchpull)
9. [背景完整同步 (watchfetch)](#9-背景完整同步-watchfetch)
10. [VTK 檔案自動重命名 (vtkrename)](#10-vtk-檔案自動重命名-vtkrename)
11. [SSH 連線與節點操作](#11-ssh-連線與節點操作)
12. [編譯與執行](#12-編譯與執行)
13. [VS Code Tasks 快捷操作](#13-vs-code-tasks-快捷操作)
14. [伺服器與節點資訊](#14-伺服器與節點資訊)
15. [同步排除規則](#15-同步排除規則)
16. [Mac/Windows 測試報告](#16-macwindows-測試報告)
17. [VPN 路由自動修復 (macOS)](#17-vpn-路由自動修復-macos)
18. [疑難排解](#18-疑難排解)

---

## 1. 環境需求與首次設定

### Windows 需求

| 工具 | 說明 |
|------|------|
| PuTTY (plink.exe / pscp.exe) | SSH 連線與檔案傳輸 |
| PowerShell 5.1+ | 執行 `mobaxterm.ps1` |

腳本：`.vscode/mobaxterm.ps1`

**首次使用** — 在 PowerShell 中執行任意指令（如 `mobaxterm help`），腳本會自動在 `$PROFILE` 建立 `mobaxterm` 函數別名。

### macOS 需求

| 工具 | 安裝方式 | 說明 |
|------|----------|------|
| ssh | 內建 | SSH 連線 |
| rsync | 內建 (`/usr/bin/rsync`) 或 `brew install rsync` | 檔案同步 |
| sshpass | `brew install hudochenkov/sshpass/sshpass` | 密碼自動輸入 |

腳本：`.vscode/cfdlab-mac.sh`

**首次設定：**

```bash
# 1. 給予執行權限
chmod +x .vscode/cfdlab-mac.sh

# 2. 安裝 sshpass（若需密碼認證）
brew install hudochenkov/sshpass/sshpass

# 3. 首次執行任意指令，腳本會自動在 ~/.profile 建立 mobaxterm 別名
.vscode/cfdlab-mac.sh help
```

**環境變數（可選）：**

```bash
export CFDLAB_PASSWORD='1256'          # 密碼（需搭配 sshpass）
export CFDLAB_ASSUME_YES=1             # 跳過確認提示
export CFDLAB_USER='chenpengchung'     # 使用者名稱
export CFDLAB_DEFAULT_NODE=3           # 預設節點
```

### 兩平台命令完全相容

Windows 和 macOS 使用**相同的命令名稱**，只是底層實作不同：

| | Windows | macOS |
|---|---|---|
| 命令前綴 | `mobaxterm <cmd>` | `mobaxterm <cmd>` |
| 底層工具 | PuTTY (plink/pscp) | rsync + ssh + sshpass |
| 腳本 | `.vscode/mobaxterm.ps1` | `.vscode/cfdlab-mac.sh` |

---

## 2. MobaXterm 同步命令總覽 (Git-like)

所有命令都以 `mobaxterm` 開頭，對應 Git 概念：

| Git 指令 | mobaxterm 指令 | 做了什麼 |
|----------|----------------|----------|
| `git status` | `mobaxterm status` | 顯示各伺服器待推送/待拉取的檔案數量 |
| `git add .` | `mobaxterm add` | 列出所有待推送的檔案名稱 |
| `git diff` | `mobaxterm diff` | 逐檔比較本地與遠端的差異（列出新增/修改/刪除的檔案） |
| `git push` | `mobaxterm push` | 將本地程式碼上傳到遠端，並刪除遠端有但本地沒有的檔案 |
| `git pull` | `mobaxterm pull` | 從遠端下載輸出檔案到本地（不刪除本地任何檔案） |
| `git fetch` | `mobaxterm fetch` | 從遠端下載到本地，並刪除本地有但遠端沒有的檔案 |
| `git log` | `mobaxterm log` | 列出遠端 log 檔案清單，並顯示最新一份的最後 20 行 |
| `git reset --hard` | `mobaxterm reset` | 刪除遠端有但本地沒有的檔案（不上傳任何東西） |
| `git clone` | `mobaxterm clone` | 從遠端完整下載所有檔案到本地（覆蓋本地） |

### 三大系列命令核心差異

```
┌──────────────────────────────────────────────────────────────┐
│  PUSH 系列    本地 ──⬆️ 上傳──→ 遠端    ⚠️ 刪除遠端多餘  │
│  PULL 系列    本地 ←──⬇️ 下載── 遠端    ✅ 不刪除本地      │
│  FETCH 系列   本地 ←──⬇️ 下載── 遠端    ⚠️ 刪除本地多餘  │
└──────────────────────────────────────────────────────────────┘
```

| 系列 | 方向 | 會刪除嗎？ | 使用時機 | 安全性 |
|------|------|-----------|----------|--------|
| **PUSH** | ⬆️ 上傳 | ⚠️ 刪除遠端多餘 | 改完程式碼要部署到伺服器 | ⚠️ |
| **PULL** | ⬇️ 下載 | ❌ 不刪除 | 安全下載模擬結果 | ✅ 最安全 |
| **FETCH** | ⬇️ 下載 | ⚠️ 刪除本地多餘 | 讓本地完全等於遠端 | ⚠️ |

---

## 3. Push 系列 — 上傳到遠端

### 流程說明

```
mobaxterm push 執行流程：
  1. 掃描本地程式碼檔案 (.h, .cu, .cpp …)
  2. 與遠端 .87 比對 → 找出 新增/修改/遠端多餘 的檔案
  3. 上傳 新增+修改 的檔案到 .87
  4. 刪除 .87 上遠端有但本地沒有的檔案
  5. 對 .154 重複步驟 2-4
  6. 輸出結果：.87: Uploaded=3 | Deleted=1 | .154: Uploaded=3 | Deleted=1
```

### 命令列表

| 命令 | 等同 | 做了什麼 |
|------|------|----------|
| `mobaxterm push` | — | 上傳本地 → 全部伺服器 (.87 + .154)，刪除遠端多餘 |
| `mobaxterm push87` | `push .87` | 只上傳到 .87 |
| `mobaxterm push89` | `push .89` | 只上傳到 .89 (Windows) |
| `mobaxterm push154` | `push .154` | 只上傳到 .154 |
| `mobaxterm pushall` | `push` | 上傳到全部伺服器 |
| `mobaxterm autopush` | — | 先檢查有無變更，有才推送（適合排程/背景使用） |
| `mobaxterm autopush87` | `autopush .87` | 只對 .87 自動推送 |
| `mobaxterm watchpush` | — | 啟動背景 daemon，持續監控本地檔案變更並自動上傳 |

### 使用範例

```bash
# 典型工作流：改完程式碼 → 推送到伺服器
mobaxterm diff          # 先看哪些檔案會被推送
mobaxterm push          # 確認後推送

# 只推送到 .87
mobaxterm push87
```

---

## 4. Pull 系列 — 從遠端下載

### 流程說明

```
mobaxterm pull 執行流程：
  1. 連接遠端 .87 伺服器
  2. 掃描遠端的輸出檔案 (*.dat, *.plt, *.vtk, *.bin, log*)
  3. 比對本地已有的檔案（MD5 hash）
  4. 只下載 新增 或 內容有變更 的檔案
  5. ✅ 不刪除本地任何檔案（即使遠端沒有）
  6. 輸出結果：3 files downloaded from .87
```

### 命令列表

| 命令 | 等同 | 做了什麼 |
|------|------|----------|
| `mobaxterm pull` | `pull .87` | 從 .87 下載輸出檔案（**不刪除**本地檔案） |
| `mobaxterm pull .87` | — | 明確指定從 .87 下載 |
| `mobaxterm pull .154` | — | 從 .154 下載 |
| `mobaxterm pull87` | `pull .87` | 快捷別名 |
| `mobaxterm pull154` | `pull .154` | 快捷別名 |
| `mobaxterm autopull` | — | 先檢查有無新檔案，有才下載 |
| `mobaxterm autopull87` | `autopull .87` | 只對 .87 自動拉取 |
| `mobaxterm watchpull` | — | 啟動背景 daemon，每 30 秒自動檢查並下載新檔案 |

### 下載的檔案類型

| 類型 | 說明 |
|------|------|
| `*.dat` / `*.DAT` | 數據輸出檔 |
| `*.plt` | Tecplot 繪圖檔 |
| `*.vtk` | VTK 視覺化檔案 |
| `*.bin` | 二進位備份檔 |
| `log*` | 執行日誌 |

### 使用範例

```bash
# 模擬跑完後，下載結果
mobaxterm pull              # 從 .87 下載
mobaxterm pull .154         # 從 .154 下載

# 背景自動下載（跑模擬時開著）
mobaxterm watchpull         # 監控兩台伺服器
mobaxterm watchpull .87     # 只監控 .87
```

---

## 5. Fetch 系列 — 完整同步（含刪除）

### 流程說明

```
mobaxterm fetch 執行流程：
  1. 連接遠端 .87 伺服器
  2. 下載遠端有的輸出檔案
  3. ⚠️ 刪除本地有但遠端沒有的輸出檔案
  4. 結果：本地輸出目錄完全等於遠端
```

### 命令列表

| 命令 | 等同 | 做了什麼 |
|------|------|----------|
| `mobaxterm fetch` | `fetch .87` | 從 .87 完整同步（下載 + **刪除本地多餘**） |
| `mobaxterm fetch .87` | — | 明確指定從 .87 同步 |
| `mobaxterm fetch .154` | — | 從 .154 同步 |
| `mobaxterm fetch87` | `fetch .87` | 快捷別名 |
| `mobaxterm fetch154` | `fetch .154` | 快捷別名 |
| `mobaxterm autofetch` | — | 先檢查有無差異，有才同步 |
| `mobaxterm watchfetch` | — | 啟動背景 daemon，每 30 秒完整同步 |

> ⚠️ **注意**：fetch 會刪除本地多餘的檔案！使用前確認本地沒有未上傳的重要資料。

---

## 6. 狀態檢查與比對

### 命令列表

| 命令 | 等同 | 做了什麼 |
|------|------|----------|
| `mobaxterm status` | — | 顯示各伺服器的待推送/待拉取的檔案數量總覽 |
| `mobaxterm diff` | `check` | 逐一列出每個伺服器上 新增/修改/刪除 的檔案名稱 |
| `mobaxterm diff87` | — | 只比對 .87 |
| `mobaxterm diff154` | — | 只比對 .154 |
| `mobaxterm diffall` | `diff` | 比對全部伺服器 |
| `mobaxterm add` | — | 列出待推送的檔案清單（類似 `git add` 後的 staging 列表） |
| `mobaxterm issynced` | — | 一行顯示同步狀態：`.87: [OK] \| .154: [DIFF]` |
| `mobaxterm log` | `log .87` | 列出 .87 的 log 檔案，並顯示最新一份的最後 20 行 |
| `mobaxterm log .154` | — | 查看 .154 的 log |
| `mobaxterm log87` | `log .87` | 快捷別名 |
| `mobaxterm log154` | `log .154` | 快捷別名 |

### 進階操作

| 命令 | 做了什麼 |
|------|----------|
| `mobaxterm sync` | 互動式同步：先顯示 diff → 問你是否確認 → 執行 push |
| `mobaxterm fullsync` | 完整同步：push + reset（讓遠端完全等於本地） |
| `mobaxterm reset` | 只刪除遠端有但本地沒有的檔案（不上傳） |
| `mobaxterm clone` | 從遠端完整複製到本地（覆蓋本地所有檔案） |

### 使用範例

```bash
# 快速查看狀態
mobaxterm issynced          # → .87: [OK] | .154: [DIFF]

# 詳細查看差異
mobaxterm diff              # 列出所有差異檔案

# 完整同步流程
mobaxterm sync              # diff → 確認 → push
```

---

## 7. 背景自動上傳 (watchpush)

### 流程說明

```
mobaxterm watchpush 執行流程：
  1. 啟動背景 daemon 程序
  2. 每 10 秒掃描本地程式碼檔案
  3. 計算 MD5 hash，與上次快照比對
  4. 若有檔案變更 → 自動上傳到 .87 和 .154
  5. 寫入日誌到 .vscode/watchpush.log
  6. 持續執行直到手動停止
```

### 命令列表

| 命令 | 做了什麼 |
|------|----------|
| `mobaxterm watchpush` | 啟動背景上傳 daemon（預設每 10 秒掃描） |
| `mobaxterm watchpush 5` | 啟動，自訂為每 5 秒掃描一次 |
| `mobaxterm watchpush status` | 顯示 daemon 是否在運行中 + PID |
| `mobaxterm watchpush log` | 顯示最近 50 行上傳日誌 |
| `mobaxterm watchpush stop` | 停止背景 daemon |
| `mobaxterm watchpush clear` | 清除日誌檔案 |

### 技術細節

- **監控間隔**：10 秒（可自訂）
- **上傳範圍**：程式碼檔案 (`.h`, `.cu`, `.c`, `.cpp` 等)
- **排除項目**：`*.dat`, `log*`, `*.plt`, `a.out`, `result/`, `backup/`
- **比對方式**：MD5 hash（只上傳內容有變更的檔案）
- **日誌位置**：`.vscode/watchpush.log`

> ✅ **安全設計**：程式輸出檔不會被上傳，避免覆蓋遠端正在寫入的資料

---

## 8. 背景自動下載 (watchpull)

### 流程說明

```
mobaxterm watchpull 執行流程：
  1. 啟動背景 daemon 程序
  2. 每 30 秒連接遠端伺服器
  3. 掃描遠端的輸出檔案 (*.dat, *.vtk, log* 等)
  4. 與本地比對 MD5 hash
  5. 若有新檔案或內容變更 → 自動下載
  6. ✅ 不會刪除本地檔案
  7. 寫入日誌到 .vscode/watchpull.log
```

### 命令列表

| 命令 | 做了什麼 |
|------|----------|
| `mobaxterm watchpull` | 啟動背景下載 daemon（監控兩台伺服器） |
| `mobaxterm watchpull .87` | 只監控 .87 |
| `mobaxterm watchpull .154` | 只監控 .154 |
| `mobaxterm watchpull status` | 顯示 daemon 是否在運行中 |
| `mobaxterm watchpull log` | 顯示最近下載日誌 |
| `mobaxterm watchpull stop` | 停止背景 daemon |
| `mobaxterm watchpull clear` | 清除日誌檔案 |

### 技術細節

- **監控間隔**：30 秒
- **下載範圍**：`*.dat`, `*.DAT`, `*.plt`, `*.bin`, `*.vtk`, `log*`
- **不會下載**：程式碼檔案 (`.h`, `.cu`, `.c` 等)
- **不會刪除**：本地任何檔案
- **日誌位置**：`.vscode/watchpull.log`

> ✅ **安全設計**：只拉取程式輸出，不會覆蓋正在編輯的程式碼

---

## 9. 背景完整同步 (watchfetch)

### 流程說明

```
mobaxterm watchfetch 執行流程：
  1. 啟動背景 daemon 程序
  2. 每 30 秒連接遠端伺服器
  3. 下載遠端有的輸出檔案
  4. ⚠️ 刪除本地有但遠端沒有的輸出檔案
  5. 結果：本地輸出目錄持續保持與遠端一致
```

### 命令列表

| 命令 | 做了什麼 |
|------|----------|
| `mobaxterm watchfetch` | 啟動背景完整同步 daemon |
| `mobaxterm watchfetch .87` | 只同步 .87 |
| `mobaxterm watchfetch .154` | 只同步 .154 |
| `mobaxterm watchfetch status` | 顯示 daemon 狀態 |
| `mobaxterm watchfetch log` | 顯示同步日誌 |
| `mobaxterm watchfetch stop` | 停止 daemon |

> ⚠️ **注意**：watchfetch 會刪除本地檔案以保持同步！

---

## 10. VTK 檔案自動重命名 (vtkrename)

### 流程說明

```
mobaxterm vtkrename 執行流程：
  1. 啟動背景 daemon 程序
  2. 每 5 秒掃描 result/ 資料夾
  3. 找出未 zero-padding 的 VTK 檔案
  4. 重命名為 6 位數 zero-padding 格式
  5. 範例：
     velocity_merged_1001.vtk    → velocity_merged_001001.vtk
     velocity_merged_31001.vtk   → velocity_merged_031001.vtk
     velocity_merged_123456.vtk  → 保持不變（已是 6 位數）
```

### 命令列表

| 命令 | 做了什麼 |
|------|----------|
| `mobaxterm vtkrename` | 啟動 VTK 重命名 daemon |
| `mobaxterm vtkrename status` | 查看是否執行中 |
| `mobaxterm vtkrename log` | 查看重命名歷史 |
| `mobaxterm vtkrename stop` | 停止 daemon |

> **用途**：確保 ParaView 能正確按時間步排序 VTK 檔案

---

## 11. SSH 連線與節點操作

### 命令列表

| 命令 | 做了什麼 |
|------|----------|
| `mobaxterm ssh 87:3` | SSH 連線到 .87 母機的 ib3 節點，進入工作目錄 |
| `mobaxterm ssh 154:4` | SSH 連線到 .154 母機的 ib4 節點 |
| `mobaxterm issh` | 互動式 SSH 選擇器：顯示 GPU 狀態 → 選擇節點 → 連線 (Windows) |
| `mobaxterm jobs 87:3` | 查看 ib3 上正在執行的 a.out 程序 |
| `mobaxterm kill 87:3` | 終止 ib3 上的執行程序 |

### GPU 狀態 (Windows 特有)

| 命令 | 做了什麼 |
|------|----------|
| `mobaxterm gpus` | 顯示所有伺服器的 GPU 使用概況 |
| `mobaxterm gpu 87` | 顯示 .87 的詳細 nvidia-smi 資訊 |

---

## 12. 編譯與執行

### 命令列表

| 命令 | 做了什麼 |
|------|----------|
| `mobaxterm run 87:3 4` | 在 .87→ib3 上編譯 main.cu 並用 4 顆 GPU 執行 |
| `mobaxterm run 154:4 8` | 在 .154→ib4 上用 8 顆 GPU 執行 |

### 編譯命令明細

```bash
cd /home/chenpengchung/D3Q27_PeriodicHill && \
nvcc main.cu -arch=sm_35 \
  -I/home/chenpengchung/openmpi-3.0.3/include \
  -L/home/chenpengchung/openmpi-3.0.3/lib \
  -lmpi -o a.out && \
nohup mpirun -np 4 ./a.out > log$(date +%Y%m%d) 2>&1 &
```

---

## 13. VS Code Tasks 快捷操作

開啟方式：`Terminal → Run Task...` 或用快捷鍵

### 快捷鍵

| 快捷鍵 | macOS 按法 | 功能 |
|--------|-----------|------|
| `Ctrl+Alt+F` | `Ctrl+Option(⌥)+F` | 切換節點 (Switch Node) |
| `Ctrl+Alt+G` | `Ctrl+Option(⌥)+G` | 重新連線 (Reconnect) |
| `Ctrl+Shift+B` | `Cmd+Shift+B` | 編譯 + 執行 (Compile + Run) |

### 跨平台任務（Mac + Windows 通用）

| 任務名稱 | 功能 |
|----------|------|
| SSH to cfdlab | 連線到伺服器節點 |
| Switch Node | 切換子機 |
| Reconnect | 重新連線 |

### Windows 專用任務（透過 PuTTY）

| 任務名稱 | 功能 |
|----------|------|
| Compile + Run (.87) / (.154) | 編譯並執行 |
| Check Running Jobs (.87) / (.154) | 查看執行中的作業 |
| Kill Running Job (.87) / (.154) | 終止作業 |
| Dashboard (.87) / (.154) | 一覽表（GPU、系統、log、磁碟） |
| GPU Status (.87) / (.154) | nvidia-smi |
| System Load (.87) / (.154) | uptime + memory + disk |
| Tail Log (.87) / (.154) | 查看最新 log |
| Disk Usage (.87) / (.154) | 磁碟使用量 |
| List Remote Files (.87) / (.154) | 遠端檔案清單 |
| Count Results (.87) / (.154) | 計算結果檔案數 |
| Clean Results (.87) / (.154) | 清除結果檔案 |
| Check All Nodes (.87) / (.154) | 檢查該母機下所有節點狀態 |

### 同步相關任務

| 任務名稱 | 功能 |
|----------|------|
| Auto Sync (Watch) | 前景自動推送 |
| Quick Sync | 有變更才推送 |
| Sync Status | 查看上傳+下載狀態 |
| Auto Upload (Start / Status / Stop) | 背景上傳管理 |
| Auto Download (Start / Status / Stop) | 背景下載管理 |
| Auto Download (.87 only) / (.154 only) | 指定伺服器下載 |

---

## 14. 伺服器與節點資訊

### 母機

| 母機 | IP | 可連節點 |
|------|-----|----------|
| .87 | 140.114.58.87 | ib2, ib3, ib5, ib6 |
| .89 | 140.114.58.89 | 直連（V100-32G × 8） |
| .154 | 140.114.58.154 | ib1, ib4, ib7, ib9 |

### 節點 GPU 配置

| 節點 | 母機 | GPU | 說明 |
|------|------|-----|------|
| .89 直連 | .89 | V100-SXM2-32GB × 8 | 最高效能 |
| ib6 | .87 | V100-SXM2-16GB × 8 | 高效能 |
| ib2, ib3, ib5 | .87 | P100-PCIE-16GB × 8 | 標準 |
| ib1, ib4, ib7, ib9 | .154 | P100-PCIE-16GB × 8 | 標準 |

### 節點狀態

| 節點 | 母機 | 狀態 |
|------|------|------|
| ib2, ib3, ib5, ib6 | .87 | ✅ 正常 |
| ib4, ib9 | .154 | ✅ 正常 |
| ib1, ib7 | .154 | ❌ 待修 |

- **工作目錄**：`/home/chenpengchung/D3Q27_PeriodicHill`
- **密碼**：`1256`

---

## 15. 同步排除規則

### Push（上傳）排除

以下檔案/資料夾**不會被上傳**到遠端：

| 排除項 | 原因 |
|--------|------|
| `.git/*` | Git 版本控制（各機獨立） |
| `.vscode/*` | VS Code 設定（本地專用） |
| `a.out` / `*.o` / `*.exe` | 編譯產物（遠端自行編譯） |
| `*.dat` / `*.DAT` / `*.plt` / `*.bin` / `*.vtk` | 模擬輸出（遠端自行產生） |
| `log*` | 執行日誌 |
| `backup/` / `result/` / `statistics/` | 輸出資料夾 |

### Pull/Fetch（下載）只包含

| 包含項 | 說明 |
|--------|------|
| `*.dat` / `*.DAT` | 數據輸出 |
| `*.plt` | Tecplot 檔案 |
| `*.bin` | 二進位備份 |
| `*.vtk` | VTK 視覺化 |
| `log*` | 執行日誌 |

---

## 16. Mac/Windows 測試報告

> 測試日期：2026-02-14

### 測試項目與結果

| # | 測試項目 | 結果 | 備註 |
|---|---------|------|------|
| 1 | 資料夾遷移（`scripts/` → `.vscode/`） | ✅ PASS | 舊 scripts 資料夾已移除 |
| 2 | Windows 相容性迴歸測試 | ✅ PASS | PS1 語法、help、bgstatus、syncstatus、check、tasks.json 全部通過 |
| 3 | 命令名稱對齊（Win vs Mac） | ✅ PASS | Mac 完全覆蓋 Windows 所有命令名稱 |
| 4 | Mac 腳本語法檢查 (`bash -n`) | ✅ PASS | 無語法錯誤 |
| 5 | Mac 本地邏輯命令 | ✅ PASS | help、bgstatus、syncstatus、issynced、check 全部正常 |
| 6 | Mac SSH 遠端連線 | ⚠️ 視環境 | 需在校內網路 + SSH key 或 sshpass |
| 7 | Mac 環境工具檢查 | ✅ PASS | rsync ✓、ssh ✓、sshpass ✓ |

### Mac 本地測試明細

```
bash -n .vscode/cfdlab-mac.sh               → PASS (語法正確)
.vscode/cfdlab-mac.sh help                  → PASS (輸出命令清單)
.vscode/cfdlab-mac.sh bgstatus              → PASS ([STOPPED] push/pull/fetch/vtkrename)
.vscode/cfdlab-mac.sh watchpush status      → PASS ([STOPPED] push daemon)
.vscode/cfdlab-mac.sh watchpull status      → PASS ([STOPPED] pull daemon)
.vscode/cfdlab-mac.sh watchfetch status     → PASS ([STOPPED] fetch daemon)
.vscode/cfdlab-mac.sh vtkrename status      → PASS ([STOPPED] vtkrename daemon)
.vscode/cfdlab-mac.sh check                 → PASS (local OK; SSH 需校內網路)
.vscode/cfdlab-mac.sh syncstatus all        → PASS (邏輯正確; SSH timeout 因非校內)
```

### Mac 環境版本

```
rsync:   openrsync protocol version 29 (/usr/bin/rsync)
ssh:     OpenSSH_9.8p1, LibreSSL 3.3.6
sshpass: 1.06 (/opt/homebrew/bin/sshpass)
```

---

## 17. VPN 路由自動修復 (macOS)

macOS VPN 連線後，`140.114.58.0/24` 子網路經常不會走 VPN 隧道，導致 SSH 連線超時。
本專案已整合三層防護：

### 自動修復機制

| 層級 | 機制 | 說明 |
|------|------|------|
| **1. LaunchDaemon** | 開機自動啟動 | 每 5 秒檢查 VPN 路由，自動修復。**完全無感。** |
| **2. cfdlab-mac.sh** | 指令前自動修復 | 執行任何遠端指令前自動檢查路由（`ensure_vpn_route`） |
| **3. 手動指令** | 終端快捷指令 | `vpnfix` / `vpncheck` / `mobaxterm vpnfix` |

### 命令列表

| 命令 | 說明 |
|------|------|
| `vpnfix` | (shell alias) 手動加入 VPN 路由 |
| `vpncheck` | (shell alias) 檢查目前路由走哪個介面 |
| `mobaxterm vpnfix` | 透過腳本修復路由，顯示詳細狀態 |

### LaunchDaemon 管理

```bash
# 查看狀態
sudo launchctl list | grep vpn

# 查看 log
cat /tmp/vpn-route-watcher.log

# 停止
sudo launchctl bootout system/com.cfdlab.vpn-route-watcher

# 重新啟動
sudo launchctl bootstrap system /Library/LaunchDaemons/com.cfdlab.vpn-route-watcher.plist
```

### 安裝檔案位置

| 檔案 | 位置 |
|------|------|
| Plist 來源 | `.vscode/com.cfdlab.vpn-route-watcher.plist` |
| 安裝位置 | `/Library/LaunchDaemons/com.cfdlab.vpn-route-watcher.plist` |
| Sudoers | `/etc/sudoers.d/vpn-route` （允許免密碼 `sudo route`） |
| Log | `/tmp/vpn-route-watcher.log` |
| Shell aliases | `~/.zshrc` 中的 `vpnfix` / `vpncheck` |

### 環境變數

```bash
export CFDLAB_VPN_AUTOFIX=0  # 設為 0 可關閉 cfdlab-mac.sh 的自動修復
```

---

## 18. 疑難排解

### Mac

| 問題 | 解決方式 |
|------|----------|
| `Missing command: rsync` | `brew install rsync` |
| `Missing command: sshpass` | `brew install hudochenkov/sshpass/sshpass` |
| SSH 連線失敗 | 確認在校內網路；手動測試 `ssh chenpengchung@140.114.58.87` |
| VPN 連上但 SSH timeout | 執行 `vpnfix` 或 `mobaxterm vpnfix` 修復路由 |
| `mobaxterm` 指令找不到 | 執行 `source ~/.profile` 或重啟終端 |
| daemon 卡住 | `mobaxterm watchpush stop && mobaxterm watchpull stop` |

### Windows

| 問題 | 解決方式 |
|------|----------|
| `plink.exe` 找不到 | 確認 PuTTY 安裝在 `C:\Program Files\PuTTY\` |
| `mobaxterm` 指令找不到 | 執行 `. $PROFILE` 或重啟 PowerShell |
| SSH 連線 timeout | 確認在校內網路或 VPN |

### 通用

| 命令 | 用途 |
|------|------|
| `mobaxterm check` | 檢查本地工具 + 遠端連線是否正常 |
| `mobaxterm bgstatus` | 查看所有背景 daemon 的狀態 |
| `mobaxterm syncstatus` | 查看同步狀態 + daemon 狀態 |

---

## 快速參考卡

```
┌─────────────────── 日常工作流程 ───────────────────┐
│                                                      │
│  改程式碼 → mobaxterm push      (上傳到伺服器)       │
│  看結果  → mobaxterm pull      (下載模擬輸出)       │
│  查狀態  → mobaxterm issynced  (一行看同步狀態)     │
│  查差異  → mobaxterm diff      (逐檔比對)           │
│                                                      │
│  ──── 背景自動化 ────                               │
│  mobaxterm watchpush           (自動上傳程式碼)     │
│  mobaxterm watchpull           (自動下載結果)       │
│  mobaxterm bgstatus            (查看全部 daemon)    │
│                                                      │
│  ──── SSH 操作 ────                                  │
│  mobaxterm ssh 87:3            (連線到 ib3)          │
│  mobaxterm run 87:3 4          (編譯 + 4 GPU 執行)  │
│  mobaxterm jobs 87:3           (查看執行中作業)      │
│  mobaxterm kill 87:3           (終止作業)            │
│                                                      │
│  ──── VPN 路由 (Mac) ────                            │
│  vpnfix                        (修復 VPN 路由)      │
│  vpncheck                      (查看路由介面)       │
│  mobaxterm vpnfix              (詳細路由修復)       │
│  ※ LaunchDaemon 開機自動修復，通常不需手動          │
│                                                      │
└──────────────────────────────────────────────────────┘
```
