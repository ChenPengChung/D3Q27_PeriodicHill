# D3Q27_PeriodicHill CFD Lab Sync Tools -- 跨平台比較分析報告

> **專案名稱:** D3Q27_PeriodicHill CFD Lab Sync Tools
> **報告日期:** 2026-02-13
> **分析版本:** Git Branch `Edit1_Re150` (Commit `53d05e1`)
> **分析範圍:** `.vscode/mobaxterm.ps1` (Windows) | `.vscode/cfdlab-mac.sh` (macOS) | `.vscode/tasks.json` (VS Code Tasks)

---

## 目錄

1. [執行摘要 (Executive Summary)](#1-執行摘要-executive-summary)
2. [測試範圍與方法 (Test Scope & Method)](#2-測試範圍與方法-test-scope--method)
3. [伺服器架構概覽 (Server Architecture)](#3-伺服器架構概覽-server-architecture)
4. [指令完整對比表 (Full Command Comparison)](#4-指令完整對比表-full-command-comparison)
5. [Tasks.json 任務對比表 (Tasks Comparison)](#5-tasksjson-任務對比表-tasks-comparison)
6. [問題清單 (Issues Found)](#6-問題清單-issues-found)
7. [建議修正方案 (Recommended Fixes)](#7-建議修正方案-recommended-fixes)
8. [結論 (Conclusion)](#8-結論-conclusion)

---

## 1. 執行摘要 (Executive Summary)

本報告針對 D3Q27_PeriodicHill 專案中用於同步本地開發環境與遠端 CFD (Computational Fluid Dynamics) 計算伺服器的跨平台工具進行完整比較分析。該工具組包含：

- **Windows 版本:** `mobaxterm.ps1` -- PowerShell 腳本，透過 PuTTY 工具組 (`pscp.exe`, `plink.exe`) 實現檔案同步與遠端操作
- **macOS 版本:** `cfdlab-mac.sh` -- Bash 腳本，透過原生 `ssh` + `rsync` + `sshpass` 實現同步
- **VS Code Tasks:** `tasks.json` -- 定義兩平台的快捷任務入口

### 關鍵發現摘要

| 指標 | Windows (`mobaxterm.ps1`) | macOS (`cfdlab-mac.sh`) |
|------|--------------------------|-------------------------|
| **支援伺服器數量** | 2 台 (.87, .154) | 3 台 (.87, .89, .154) |
| **獨立指令總數** | ~27 個 | ~65 個 |
| **同步引擎** | PuTTY (`pscp`/`plink`) + MD5 逐檔比較 | `rsync` (差異壓縮傳輸) |
| **背景 Daemon 種類** | 4 種 (watchpush/pull/fetch/vtkrename) | 4 種 (同上) |
| **GPU 監控** | 無 | 完整 (`gpus`, `gpu [server]`) |
| **SSH/遠端執行** | 僅透過 tasks.json hardcoded | 腳本內建 (`ssh`, `run`, `jobs`, `kill`, `issh`) |
| **環境檢查** | 無 | 有 (`check` 指令) |

**結論:** macOS 版本在功能覆蓋率、伺服器支援範圍及操作便利性上顯著領先 Windows 版本。Windows 版本缺少對最強運算伺服器 .89 的支援，且缺少大量伺服器特定快捷指令和 GPU 監控功能。此外，兩平台的 `check` 指令存在語意衝突，可能導致使用者混淆。

---

## 2. 測試範圍與方法 (Test Scope & Method)

### 2.1 分析檔案

| 檔案路徑 | 類型 | 大小 (行數) | 角色 |
|----------|------|------------|------|
| `.vscode/mobaxterm.ps1` | PowerShell | ~1,549 行 | Windows 平台主腳本 |
| `.vscode/cfdlab-mac.sh` | Bash | ~1,453 行 | macOS 平台主腳本 |
| `.vscode/tasks.json` | JSON | ~1,516 行 | VS Code 任務定義 (雙平台共用) |
| `.vscode/ssh-connect.ps1` | PowerShell | (參照) | Windows SSH 連線輔助腳本 |
| `.vscode/watchpush-daemon.ps1` | PowerShell | (參照) | Windows watchpush 背景 daemon |
| `.vscode/watchpull-daemon.ps1` | PowerShell | (參照) | Windows watchpull 背景 daemon |
| `.vscode/watchfetch-daemon.ps1` | PowerShell | (參照) | Windows watchfetch 背景 daemon |

### 2.2 分析方法

1. **靜態程式碼分析:** 逐行閱讀兩平台腳本的 `switch` / `case` 分支，列舉所有可用指令
2. **組態比較:** 比對 `$Config.Servers` (Windows) 與 `normalize_server()` / `resolve_host()` (macOS) 的伺服器定義
3. **Tasks.json 交叉比對:** 分析每個 task 的 `windows` / `osx` / `linux` 欄位，識別單平台任務
4. **語意比較:** 針對同名指令，比較實際功能邏輯是否一致

---

## 3. 伺服器架構概覽 (Server Architecture)

### 3.1 伺服器清單

| 伺服器 | IP 位址 | 連線方式 | GPU 配置 | Windows 支援 | macOS 支援 |
|--------|---------|----------|----------|:------------:|:----------:|
| **.87** | 140.114.58.87 | 跳板機 (gateway) | ib2/ib3/ib5: 8x P100-16GB; ib6: 8x V100-16GB | YES | YES |
| **.89** | 140.114.58.89 | 直連 (direct) | 8x V100-SXM2-**32GB** | **NO** | YES |
| **.154** | 140.114.58.154 | 跳板機 (gateway) | ib1/ib4/ib7/ib9: 8x P100-16GB | YES | YES |

### 3.2 Windows 伺服器組態 (mobaxterm.ps1 第 23-26 行)

```powershell
$script:Config = @{
    Servers = @(
        @{ Name = ".87";  Host = "140.114.58.87";  User = "chenpengchung"; Password = "1256" },
        @{ Name = ".154"; Host = "140.114.58.154"; User = "chenpengchung"; Password = "1256" }
    )
    # 注意: .89 伺服器完全不在此列表中
}
```

### 3.3 macOS 伺服器組態 (cfdlab-mac.sh 第 72-88 行)

```bash
function normalize_server() {
  local raw="${1:-87}"
  raw="${raw#.}"
  case "$raw" in
    87|89|154) echo "$raw" ;;  # 三台伺服器均支援
    *) die "Unknown server '$1' (use 87, 89 or 154)" ;;
  esac
}

function resolve_host() {
  case "$server" in
    87)  echo "140.114.58.87"  ;;
    89)  echo "140.114.58.89"  ;;
    154) echo "140.114.58.154" ;;
  esac
}
```

### 3.4 同步引擎差異

| 面向 | Windows (PuTTY) | macOS (rsync) |
|------|-----------------|---------------|
| **檔案比較** | 逐檔 MD5 hash (遠端 `find ... -exec md5sum`)，再逐檔 pscp | `rsync --dry-run --itemize-changes` |
| **傳輸機制** | `pscp.exe` 單檔 SCP | `rsync -az` 差異壓縮傳輸 |
| **刪除機制** | 逐檔 `plink rm -f` | `rsync --delete` |
| **效能** | O(n) 連線次數 (每檔一次 pscp) | O(1) 連線次數 (單次 rsync) |
| **Push 排除清單** | `.git/*`, `.vscode/*`, `a.out`, `*.o`, `*.exe` | `.git/`, `.vscode/`, `backup/`, `result/`, `a.out`, `*.o`, `*.exe`, `*.dat`, `*.DAT`, `*.plt`, `*.bin`, `*.vtk`, `log*` |
| **Pull 篩選** | 下載遠端有、本地沒有或不同的所有檔案 | 僅下載 `*.dat`, `*.DAT`, `*.plt`, `*.bin`, `*.vtk`, `log*` |

---

## 4. 指令完整對比表 (Full Command Comparison)

### 4.1 核心同步指令 (Core Sync Commands)

| 指令名稱 | Windows | macOS | 功能說明 | 備註 |
|----------|:-------:|:-----:|----------|------|
| `diff` | YES | YES | 比較本地與遠端檔案差異 | Windows 對所有 servers 迴圈; macOS 可指定 server 或 all |
| `check` | YES | YES | **功能完全不同!** | Windows: `diff` 的別名; macOS: 環境驗證 (ssh/rsync/sshpass 檢查 + 連線測試) |
| `status` | YES | YES | 顯示同步狀態總覽 | 功能一致 |
| `add` | YES | YES | 顯示待上傳的變更清單 | 功能一致 |
| `push` | YES | YES | 上傳本地檔案至遠端 + 刪除遠端多餘檔案 | Windows 固定對兩台; macOS 可指定 server 或 all (預設 all) |
| `pull` | YES | YES | 從遠端下載檔案 (不刪除本地) | 功能一致 |
| `fetch` | YES | YES | 從遠端下載 + 刪除本地多餘檔案 | 功能一致 |
| `log` | YES | YES | 查看遠端 log 檔案 | 功能一致 |
| `reset` / `delete` | YES | YES | 刪除遠端多餘檔案 | Windows 需互動確認; macOS 透過 `CFDLAB_ASSUME_YES` 控制 |
| `clone` | YES | YES | 從遠端完整下載覆蓋本地 | 功能一致 |
| `sync` | YES | YES | 互動式 diff + push | 功能一致 |
| `fullsync` | YES | YES | push + reset (強制同步) | 功能一致 |
| `issynced` | YES | YES | 快速同步狀態檢查 (one-line) | Windows 只檢查 .87/.154; macOS 檢查 .87/.89/.154 |

### 4.2 伺服器特定快捷指令 (Server-Specific Shortcuts)

| 指令名稱 | Windows | macOS | 備註 |
|----------|:-------:|:-----:|------|
| `pull87` | YES | YES | |
| `pull89` | **NO** | YES | Windows 缺少 .89 |
| `pull154` | YES | YES | |
| `fetch87` | YES | YES | |
| `fetch89` | **NO** | YES | Windows 缺少 .89 |
| `fetch154` | YES | YES | |
| `push87` | **NO** | YES | Windows 只有一個 `push` 指令，固定推送至所有 server |
| `push89` | **NO** | YES | Windows 缺少 .89 |
| `push154` | **NO** | YES | Windows 無伺服器特定 push |
| `pushall` | **NO** | YES | macOS 明確的 push 至全部三台 |

### 4.3 Auto 系列指令 (Auto Commands -- Single Execution)

| 指令名稱 | Windows | macOS | 備註 |
|----------|:-------:|:-----:|------|
| `autopush` | YES | YES | Windows 固定推全部; macOS 可指定 target |
| `autopush87` | **NO** | YES | |
| `autopush89` | **NO** | YES | |
| `autopush154` | **NO** | YES | |
| `autopushall` | **NO** | YES | |
| `autopull` | YES | YES | Windows 預設 .87 可帶參數; macOS 同 |
| `autopull87` | **NO** | YES | |
| `autopull89` | **NO** | YES | |
| `autopull154` | **NO** | YES | |
| `autofetch` | YES | YES | |
| `autofetch87` | **NO** | YES | |
| `autofetch89` | **NO** | YES | |
| `autofetch154` | **NO** | YES | |

### 4.4 Watch 系列指令 (Background Daemon Commands)

| 指令名稱 | Windows | macOS | 備註 |
|----------|:-------:|:-----:|------|
| `watch` | YES | YES | 啟動 watchpush |
| `watchpush [start]` | YES | YES | |
| `watchpush stop` | YES | YES | |
| `watchpush status` | YES | YES | |
| `watchpush log` | YES | YES | |
| `watchpush clear` | YES | YES | |
| `watchpull [server]` | YES (.87/.154) | YES (.87/.89/.154/all) | Windows 不支援 .89 |
| `watchpull stop` | YES | YES | |
| `watchpull status` | YES | YES | |
| `watchpull log` | YES | YES | |
| `watchpull clear` | YES | YES | |
| `watchfetch [server]` | YES (.87/.154) | YES (.87/.89/.154) | Windows 不支援 .89 |
| `watchfetch stop` | YES | YES | |
| `watchfetch status` | YES | YES | |
| `watchfetch log` | YES | YES | |
| `watchfetch clear` | YES | YES | |
| `vtkrename [start]` | YES | YES | |
| `vtkrename stop` | YES | YES | |
| `vtkrename status` | YES | YES | |
| `vtkrename log` | YES | YES | |
| `vtkrename clear` | YES | YES | |

### 4.5 狀態查詢指令 (Status Commands)

| 指令名稱 | Windows | macOS | 備註 |
|----------|:-------:|:-----:|------|
| `bgstatus` | YES | YES | 所有 daemon 狀態 |
| `syncstatus` | YES | YES | push + pull 狀態 |

### 4.6 Diff 系列指令 (比較用)

| 指令名稱 | Windows | macOS | 備註 |
|----------|:-------:|:-----:|------|
| `diff` | YES | YES | |
| `diff87` | **NO** | YES | |
| `diff89` | **NO** | YES | |
| `diff154` | **NO** | YES | |
| `diffall` | **NO** | YES | |

### 4.7 Log 系列指令

| 指令名稱 | Windows | macOS | 備註 |
|----------|:-------:|:-----:|------|
| `log` | YES | YES | |
| `log87` | **NO** | YES | |
| `log89` | **NO** | YES | |
| `log154` | **NO** | YES | |

### 4.8 GPU 監控指令 (GPU Monitoring)

| 指令名稱 | Windows | macOS | 備註 |
|----------|:-------:|:-----:|------|
| `gpus` | **NO** | YES | 所有伺服器 GPU 使用率總覽 (含即時查詢) |
| `gpu [server]` | **NO** | YES | 指定伺服器的 `nvidia-smi` 完整輸出 |

### 4.9 SSH / 遠端執行指令 (Remote Execution)

| 指令名稱 | Windows (腳本內) | macOS (腳本內) | 備註 |
|----------|:----------------:|:--------------:|------|
| `ssh [server:node]` | **NO** | YES | 互動式 SSH 連線 (含 GPU 狀態顯示) |
| `issh` | **NO** | YES | 互動式選單: 顯示所有節點 GPU 狀態, 選擇後自動 SSH |
| `run [server:node] [gpu]` | **NO** | YES | 遠端編譯 + 執行 CUDA 程式 |
| `jobs [server:node]` | **NO** | YES | 查看遠端執行中的 process |
| `kill [server:node]` | **NO** | YES | 終止遠端執行中的 process |

> **注意:** Windows 的 SSH / Compile / Jobs / Kill 功能透過 `tasks.json` 中 hardcoded 的 `plink.exe` 呼叫實現，而非 `mobaxterm.ps1` 腳本內部。macOS 則統一在腳本內實現。

### 4.10 指令統計總表

| 類別 | Windows 數量 | macOS 數量 | 差異 |
|------|:-----------:|:----------:|:----:|
| 核心同步 (push/pull/fetch/diff/log) | 11 | 28 | -17 |
| Auto 系列 | 3 | 13 | -10 |
| Watch 系列 (含子指令) | 21 | 21 | 0 |
| 狀態查詢 | 3 | 3 | 0 |
| GPU 監控 | 0 | 2 | -2 |
| SSH / 遠端執行 | 0 | 5 | -5 |
| 其他 (clone/reset/sync/fullsync) | 5 | 5 | 0 |
| **總計** | **~43** | **~77** | **-34** |

---

## 5. Tasks.json 任務對比表 (Tasks Comparison)

### 5.1 雙平台共用任務 (Cross-Platform Tasks)

以下任務同時定義了 `windows` 和 `osx`/`linux` 的 command：

| Task Label | Windows 實作 | macOS 實作 | 備註 |
|------------|-------------|------------|------|
| SSH to cfdlab | `ssh-connect.ps1` | `cfdlab-mac.sh ssh` | 均支援 server:node 選擇 |
| Reconnect | `ssh-connect.ps1` | `cfdlab-mac.sh ssh` | 功能與 SSH 相同 |
| Switch Node | `ssh-connect.ps1` | `cfdlab-mac.sh issh` | macOS 使用互動式選單 (issh) |

### 5.2 Windows 專屬任務 (Windows-Only Tasks)

| Task Label | 實作方式 | 功能 | 是否有 macOS 對應腳本指令 |
|------------|---------|------|:------------------------:|
| SSH to cfdlab (.87) | `plink.exe` hardcoded | SSH 至 .87 指定節點 | YES (腳本內 `ssh 87:N`) |
| SSH to cfdlab (.154) | `plink.exe` hardcoded | SSH 至 .154 指定節點 | YES (腳本內 `ssh 154:N`) |
| Switch Node (.87) | `plink.exe` hardcoded | 切換 .87 節點 | YES (腳本內 `issh`) |
| Switch Node (.154) | `plink.exe` hardcoded | 切換 .154 節點 | YES (腳本內 `issh`) |
| Reconnect (.87) | `plink.exe` hardcoded | 重連 .87 | YES (腳本內 `ssh`) |
| Reconnect (.154) | `plink.exe` hardcoded | 重連 .154 | YES (腳本內 `ssh`) |
| Compile + Run (.87) | `plink.exe` hardcoded | 遠端編譯 + 執行 | YES (腳本內 `run 87:N`) |
| Compile + Run (.154) | `plink.exe` hardcoded | 遠端編譯 + 執行 | YES (腳本內 `run 154:N`) |
| Check Running Jobs (.87) | `plink.exe` hardcoded | 查看 .87 上的 process | YES (腳本內 `jobs`) |
| Check Running Jobs (.154) | `plink.exe` hardcoded | 查看 .154 上的 process | YES (腳本內 `jobs`) |
| Kill Running Job (.87) | `plink.exe` hardcoded | 終止 .87 上的 process | YES (腳本內 `kill`) |
| Kill Running Job (.154) | `plink.exe` hardcoded | 終止 .154 上的 process | YES (腳本內 `kill`) |
| Auto Sync (Watch) | `mobaxterm.ps1 watch` | 啟動 watchpush | YES (macOS 有對應 task) |
| Quick Sync (Push if changed) | `mobaxterm.ps1 autopush` | 快速推送 | YES (macOS 有對應 task) |
| Build main.exe (MinGW) | `g++.exe` (Strawberry) | 本地 C++ 編譯 | NO (macOS 不需要本地編譯) |
| C/C++: g++.exe build active file | `g++.exe` (Strawberry) | 編譯當前檔案 | NO (同上) |

### 5.3 macOS 專屬任務 ([Mac] Prefix Tasks)

| Task Label | 腳本指令 | 功能 | Windows 是否有對應 |
|------------|---------|------|:-----------------:|
| [Mac] Check Environment | `check` | 驗證 SSH/rsync/sshpass 環境 | **NO** |
| [Mac] SSH to cfdlab | `ssh` | SSH 連線 | YES (plink task) |
| [Mac] Compile + Run | `run` | 遠端編譯 + 執行 | YES (plink task) |
| [Mac] Check Running Jobs | `jobs` | 查看執行中 process | YES (plink task) |
| [Mac] Kill Running Job | `kill` | 終止 process | YES (plink task) |
| [Mac] Sync Status | `syncstatus` | 同步狀態 | YES (Windows task) |
| [Mac] Auto Pull (once) | `autopull` | 單次自動下載 | **NO** (Windows 無 task) |
| [Mac] Auto Push (once) | `autopush` | 單次自動上傳 | YES (Quick Sync task) |
| [Mac] Watch Pull | `watchpull` | 背景自動下載 | YES (Auto Download Start) |
| [Mac] Watch Push | `watchpush` | 背景自動上傳 | YES (Auto Upload Start) |
| [Mac] Quick Sync (Push if changed) | `autopush` | 快速推送 | YES (Quick Sync task) |
| [Mac] Auto Download (Start) | `watchpull` | 啟動背景下載 | YES |
| [Mac] Auto Download (.87 only) | `watchpull 87` | 背景下載 .87 | YES |
| [Mac] Auto Download (.154 only) | `watchpull 154` | 背景下載 .154 | YES |
| [Mac] Auto Download (.89 only) | `watchpull 89` | 背景下載 .89 | **NO** |
| [Mac] Auto Download (Status) | `watchpull status` | 查看下載狀態 | YES |
| [Mac] Auto Download (Stop) | `watchpull stop` | 停止背景下載 | YES |
| [Mac] Auto Upload (Start) | `watchpush` | 啟動背景上傳 | YES |
| [Mac] Auto Upload (Status) | `watchpush status` | 查看上傳狀態 | YES |
| [Mac] Auto Upload (Stop) | `watchpush stop` | 停止背景上傳 | YES |
| [Mac] Sync Status (Upload + Download) | `syncstatus` | 綜合狀態 | YES |
| [Mac] Background Status (All) | `bgstatus` | 所有 daemon 狀態 | **NO** (Windows 無 task) |
| [Mac] GPU Status (All Servers) | `gpus` | GPU 總覽 | **NO** |
| [Mac] GPU Detail (.89) | `gpu 89` | .89 GPU 詳情 | **NO** |
| [Mac] GPU Detail (.87) | `gpu 87` | .87 GPU 詳情 | **NO** |
| [Mac] GPU Detail (.154) | `gpu 154` | .154 GPU 詳情 | **NO** |
| [Mac] Diff (Compare local vs remote) | `diff` | 比較差異 | **NO** (Windows 無 task) |
| [Mac] Push (Upload + Delete) | `push` | 推送 + 刪除 | **NO** (Windows 無 task) |
| [Mac] Pull (.87) | `pull 87` | 從 .87 下載 | **NO** |
| [Mac] Pull (.154) | `pull 154` | 從 .154 下載 | **NO** |
| [Mac] Pull (.89) | `pull 89` | 從 .89 下載 | **NO** |
| [Mac] Fetch (.87) | `fetch 87` | 同步 .87 | **NO** |
| [Mac] Fetch (.154) | `fetch 154` | 同步 .154 | **NO** |
| [Mac] Fetch (.89) | `fetch 89` | 同步 .89 | **NO** |
| [Mac] Watch Fetch (.89 only) | `watchfetch 89` | 背景同步 .89 | **NO** |
| [Mac] Watch Fetch (Start) | `watchfetch` | 啟動背景 fetch | **NO** |
| [Mac] Watch Fetch (.87 only) | `watchfetch 87` | 背景同步 .87 | **NO** |
| [Mac] Watch Fetch (.154 only) | `watchfetch 154` | 背景同步 .154 | **NO** |
| [Mac] Watch Fetch (Status) | `watchfetch status` | 查看 fetch 狀態 | **NO** |
| [Mac] Watch Fetch (Stop) | `watchfetch stop` | 停止背景 fetch | **NO** |
| [Mac] Log (Remote) | `log` | 查看遠端 log | **NO** (Windows 無 task) |
| [Mac] Is Synced (Quick Check) | `issynced` | 快速同步檢查 | **NO** |
| [Mac] Auto Fetch (once) | `autofetch` | 單次自動 fetch | **NO** |
| [Mac] VTK Rename (Start) | `vtkrename` | 啟動 VTK 重新命名 | **NO** |
| [Mac] VTK Rename (Status) | `vtkrename status` | 查看重命名狀態 | **NO** |
| [Mac] VTK Rename (Stop) | `vtkrename stop` | 停止重命名 | **NO** |
| [Mac] Auto Sync (Watch) | `watchpush` | 背景 watchpush | YES (Auto Sync Watch) |

### 5.4 任務數量統計

| 類別 | 數量 |
|------|:----:|
| 雙平台共用 (cross-platform definition) | 3 |
| Windows 專屬任務 (含 plink hardcoded) | 16 |
| macOS 專屬任務 ([Mac] prefix) | 45 |
| **tasks.json 總任務數** | **64** |

---

## 6. 問題清單 (Issues Found)

### 6.1 嚴重問題 (CRITICAL)

#### ISSUE-001: `check` 指令命名衝突 (Naming Conflict)

| 屬性 | 說明 |
|------|------|
| **嚴重度** | CRITICAL |
| **影響** | 跨平台使用者行為不一致，可能造成誤操作 |
| **檔案** | `mobaxterm.ps1` 第 165 行; `cfdlab-mac.sh` 第 1343-1360 行 |

**詳細說明:**

- **Windows (`mobaxterm.ps1`):** `check` 是 `diff` 的別名，執行「比較本地與遠端檔案差異」
  ```powershell
  { $_ -in "diff", "check" } {
      Write-Color "[DIFF] Comparing local vs remote..." "Magenta"
  ```

- **macOS (`cfdlab-mac.sh`):** `check` 是「環境驗證」指令，檢查 `ssh`、`rsync`、`sshpass` 工具是否安裝，並測試 SSH 連線
  ```bash
  function cmd_check() {
      require_cmd ssh
      require_cmd rsync
      ensure_password_tooling
      echo "[CHECK] local commands OK (ssh, rsync)"
      for server in 87 154; do
          # ... 測試 SSH 連線
      done
  }
  ```

**風險:** 慣用 macOS 的使用者在 Windows 上執行 `check` 時，預期看到環境檢查結果，卻會觸發耗時的遠端檔案差異比較。

---

#### ISSUE-002: .89 伺服器在 Windows 完全缺失

| 屬性 | 說明 |
|------|------|
| **嚴重度** | CRITICAL |
| **影響** | Windows 使用者無法操作最強運算資源 |
| **檔案** | `mobaxterm.ps1` 第 20-27 行 ($Config.Servers) |

**詳細說明:**

`.89` (140.114.58.89) 配備 **8x Tesla V100-SXM2-32GB**，是三台伺服器中計算能力最強的。然而 Windows 的 `$Config.Servers` 陣列僅包含 `.87` 和 `.154`，導致：

- 無法對 .89 執行 push / pull / fetch
- watchpull / watchfetch 無法監控 .89
- autopush / autopull / autofetch 不作用於 .89
- `issynced` 只檢查兩台而非三台
- 所有依賴 `foreach ($server in $Config.Servers)` 的操作均跳過 .89

---

### 6.2 高優先度問題 (HIGH)

#### ISSUE-003: Windows 缺少 34+ 個伺服器特定快捷指令

| 屬性 | 說明 |
|------|------|
| **嚴重度** | HIGH |
| **影響** | Windows 使用者操作效率低，無法快速指定伺服器 |
| **缺少指令** | 見下表 |

缺少的指令完整清單：

| 類別 | macOS 有但 Windows 缺少的指令 |
|------|------------------------------|
| Push 快捷 | `push87`, `push89`, `push154`, `pushall` |
| Pull 快捷 | `pull89` |
| Fetch 快捷 | `fetch89` |
| AutoPush | `autopush87`, `autopush89`, `autopush154`, `autopushall` |
| AutoPull | `autopull87`, `autopull89`, `autopull154` |
| AutoFetch | `autofetch87`, `autofetch89`, `autofetch154` |
| Diff 快捷 | `diff87`, `diff89`, `diff154`, `diffall` |
| Log 快捷 | `log87`, `log89`, `log154` |

---

#### ISSUE-004: tasks.json 中 hardcoded 路徑

| 屬性 | 說明 |
|------|------|
| **嚴重度** | HIGH |
| **影響** | 其他 Windows 使用者 (不同使用者名稱 / 磁碟路徑) 無法使用這些 task |
| **檔案** | `tasks.json` 第 323 行, 第 337 行 |

**問題程式碼:**

```json
"command": "& 'c:\\Users\\88697.CHENPENGCHUNG12\\Desktop\\GitHub-PeriodicHill\\D3Q27_PeriodicHill\\.vscode\\mobaxterm.ps1' watch"
```

```json
"command": "& 'c:\\Users\\88697.CHENPENGCHUNG12\\Desktop\\GitHub-PeriodicHill\\D3Q27_PeriodicHill\\.vscode\\mobaxterm.ps1' autopush"
```

**應修改為:**

```json
"command": "& '${workspaceFolder}\\.vscode\\mobaxterm.ps1' watch"
```

> 同一檔案中的其他 Windows task 已正確使用 `${workspaceFolder}`，只有這兩個 task 使用了寫死路徑。

---

### 6.3 中等問題 (MEDIUM)

#### ISSUE-005: Windows 缺少 GPU 監控功能

| 屬性 | 說明 |
|------|------|
| **嚴重度** | MEDIUM |
| **影響** | Windows 使用者無法在同步工具中查看 GPU 狀態，需要另外手動 SSH 檢查 |
| **缺少指令** | `gpus`, `gpu [server]` |

macOS 的 `gpus` 指令提供精美的表格化 GPU 使用率總覽 (包含即時並行查詢所有節點)，`gpu [server]` 提供完整 `nvidia-smi` 輸出。Windows 完全缺少此功能。

---

#### ISSUE-006: Windows 缺少腳本內建 SSH / 遠端執行指令

| 屬性 | 說明 |
|------|------|
| **嚴重度** | MEDIUM |
| **影響** | Windows 的 SSH / run / jobs / kill 功能依賴 tasks.json 中的 hardcoded plink 路徑，非腳本化且難以維護 |
| **缺少指令** | `ssh`, `issh`, `run`, `jobs`, `kill` |

macOS 將這些功能整合在 `cfdlab-mac.sh` 腳本中，提供統一的參數解析 (`server:node` 格式)、自動建立遠端目錄、GPU 狀態預覽等附加功能。Windows 的 tasks.json 實作缺少這些增強功能。

---

#### ISSUE-007: 同步引擎效能與排除規則差異

| 屬性 | 說明 |
|------|------|
| **嚴重度** | MEDIUM |
| **影響** | Push 排除清單不一致可能導致意外同步大型資料檔案; Windows 效能顯著低於 macOS |

**排除清單差異:**

macOS 額外排除而 Windows 未排除的檔案類型：
- `backup/` 整個目錄
- `result/` 整個目錄
- `statistics/` 整個目錄 (Windows watch 中排除但 push 中未排除)
- `*.dat`, `*.DAT`, `*.plt`, `*.bin`, `*.vtk`
- `log*`

這表示 Windows 的 `push` 可能會上傳大量模擬結果資料檔至遠端，而 macOS 不會。

---

#### ISSUE-008: macOS `check` 只測試 .87 和 .154，未測試 .89

| 屬性 | 說明 |
|------|------|
| **嚴重度** | MEDIUM |
| **影響** | `check` 環境驗證不完整 |
| **��案** | `cfdlab-mac.sh` 第 1351 行 |

```bash
for server in 87 154; do   # 應為 87 89 154
```

---

### 6.4 低優先度問題 (LOW)

#### ISSUE-009: Windows `$Config.LocalPath` 寫死

| 屬性 | 說明 |
|------|------|
| **嚴重度** | LOW |
| **影響** | 專案移動路徑後需手動修改腳本 |
| **檔案** | `mobaxterm.ps1` 第 21 行 |

```powershell
LocalPath = "c:\Users\88697.CHENPENGCHUNG12\Desktop\GitHub-PeriodicHill\D3Q27_PeriodicHill"
```

macOS 版本透過 `SCRIPT_DIR` / `WORKSPACE_DIR` 動態推導，不受路徑變更影響：
```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
```

---

#### ISSUE-010: Windows `watch` 只監控特定副檔名

| 屬性 | 說明 |
|------|------|
| **嚴重度** | LOW |
| **影響** | 新增的檔案類型可能不會被自動同步 |
| **檔案** | `mobaxterm.ps1` 第 619 行 |

```powershell
$syncExtensions = @(".cu", ".h", ".c", ".json", ".md", ".txt", ".ps1")
```

macOS 的 `watchpush` 使用 rsync 差異比較，會同步所有非排除的檔案，不受副檔名限制。

---

#### ISSUE-011: tasks.json 中密碼明文暴露

| 屬性 | 說明 |
|------|------|
| **嚴重度** | LOW (考量為內部實驗室環境) |
| **影響** | 安全性風險 |
| **檔案** | `tasks.json` 多處, `mobaxterm.ps1` 第 24-25 行 |

所有 plink/pscp 呼叫均以明文方式傳遞密碼 (`-pw 1256`)。macOS 版本透過環境變數 `CFDLAB_PASSWORD` 傳遞密碼並配合 `sshpass`，同樣非最佳實踐，但至少支援透過 SSH key 免密碼模式。

---

## 7. 建議修正方案 (Recommended Fixes)

### 優先順序 1: 緊急修正 (Immediate Fixes)

#### Fix-1: 解決 `check` 命名衝突
**建議:** 統一 `check` 語意為「環境驗證」(與 macOS 一致)，Windows 的 `diff` 別名改為其他名稱或直接移除 `check` 別名。

```powershell
# mobaxterm.ps1 -- 移除 check 作為 diff 別名
# 修改前:
{ $_ -in "diff", "check" } { ... }

# 修改後:
"diff" { ... }

# 新增獨立的 check 指令 (環境檢查):
"check" {
    Write-Color "[CHECK] Verifying environment..." "Magenta"
    # 檢查 pscp.exe, plink.exe 是否存在
    # 測試 SSH 連線
}
```

#### Fix-2: 新增 .89 伺服器至 Windows
**建議:** 在 `$Config.Servers` 新增 .89 伺服器定義。

```powershell
$script:Config = @{
    Servers = @(
        @{ Name = ".87";  Host = "140.114.58.87";  User = "chenpengchung"; Password = "1256" },
        @{ Name = ".89";  Host = "140.114.58.89";  User = "chenpengchung"; Password = "1256" },
        @{ Name = ".154"; Host = "140.114.58.154"; User = "chenpengchung"; Password = "1256" }
    )
}
```

> 注意: .89 為直連模式 (不需跳板機)，需確認 `plink`/`pscp` 連線邏輯不受影響。

#### Fix-3: 修正 tasks.json hardcoded 路徑
**建議:** 將第 323 行和第 337 行的寫死路徑改為 `${workspaceFolder}`。

---

### 優先順序 2: 功能增強 (Feature Parity)

#### Fix-4: 新增 Windows 伺服器特定快捷指令
**建議:** 在 `mobaxterm.ps1` 的 `switch` 區塊新增以下指令：

- `push87`, `push89`, `push154`, `pushall`
- `pull89`
- `fetch89`
- `autopush87/89/154/all`, `autopull87/89/154`, `autofetch87/89/154`
- `diff87/89/154/diffall`
- `log87/89/154`

模式參考現有的 `pull87` / `pull154` / `fetch87` / `fetch154` 實作。

#### Fix-5: 新增 Windows GPU 監控指令
**建議:** 在 `mobaxterm.ps1` 新增 `gpus` 和 `gpu` 指令，透過 `plink.exe` 執行遠端 `nvidia-smi`。

#### Fix-6: 將 Windows SSH/Run/Jobs/Kill 整合至 mobaxterm.ps1
**建議:** 參照 macOS 的 `cmd_ssh()`, `cmd_run()`, `cmd_jobs()`, `cmd_kill()` 實作，在 `mobaxterm.ps1` 中新增對應指令，取代 tasks.json 中的 hardcoded plink 呼叫。

---

### 優先順序 3: 品質改善 (Quality Improvements)

#### Fix-7: 統一 Push 排除清單
**建議:** 將 Windows 的 `$Config.ExcludePatterns` 擴充至與 macOS 一致：

```powershell
ExcludePatterns = @(
    ".git/*", ".vscode/*", "backup/*", "result/*", "statistics/*",
    "a.out", "*.o", "*.exe", "*.dat", "*.DAT", "*.plt", "*.bin", "*.vtk", "log*"
)
```

#### Fix-8: 動態推導 LocalPath
**建議:** 改用 `$PSScriptRoot` 動態推導，與 macOS 行為一致：

```powershell
$script:Config = @{
    LocalPath = (Split-Path $PSScriptRoot -Parent)
    # ...
}
```

#### Fix-9: macOS `check` 加入 .89 連線測試
**建議:** 修改 `cmd_check()` 的 server loop：

```bash
for server in 87 89 154; do
```

---

## 8. 結論 (Conclusion)

### 8.1 總體評估

D3Q27_PeriodicHill CFD Lab Sync Tools 在 macOS 和 Windows 兩平台間存在**顯著的功能不對稱性**。macOS 版本 (`cfdlab-mac.sh`) 提供了較為完整、結構化的功能集，涵蓋三台伺服器的同步操作、GPU 監控、互動式 SSH 連線選單等進階功能。Windows 版本 (`mobaxterm.ps1`) 缺少對 .89 伺服器的支援，且大量操作需要依賴 tasks.json 中寫死路徑的 plink 呼叫來補足。

### 8.2 風險矩陣

| 問題編號 | 嚴重度 | 影響面 | 修復難度 | 建議優先順序 |
|:--------:|:------:|:------:|:--------:|:----------:|
| ISSUE-001 | CRITICAL | 兩平台使用者 | 低 | P0 |
| ISSUE-002 | CRITICAL | Windows 使用者 | 低 | P0 |
| ISSUE-004 | HIGH | Windows 可攜性 | 低 | P0 |
| ISSUE-003 | HIGH | Windows 操作效率 | 中 | P1 |
| ISSUE-005 | MEDIUM | Windows 使用者 | 中 | P1 |
| ISSUE-006 | MEDIUM | Windows 維護性 | 高 | P2 |
| ISSUE-007 | MEDIUM | 資料完整性 | 低 | P1 |
| ISSUE-008 | MEDIUM | macOS 檢查完整性 | 低 | P0 |
| ISSUE-009 | LOW | Windows 可攜性 | 低 | P1 |
| ISSUE-010 | LOW | Windows 自動同步覆蓋率 | 低 | P2 |
| ISSUE-011 | LOW | 安全性 | 中 | P3 |

### 8.3 建議行動

1. **立即:** 修正 `check` 命名衝突、新增 .89 伺服器至 Windows、修正 hardcoded 路徑 (ISSUE-001, 002, 004)
2. **短期:** 補齊 Windows 快捷指令、統一排除清單、修正 macOS check 範圍 (ISSUE-003, 007, 008)
3. **中期:** 新增 Windows GPU 監控、整合 SSH 指令至腳本 (ISSUE-005, 006)
4. **長期:** 動態路徑推導、檔案類型自動偵測、考慮 SSH key 認證替代明文密碼 (ISSUE-009, 010, 011)

---

> **報告結束**
> 產生工具: Claude Code (claude-opus-4-6)
> 報告日期: 2026-02-13
