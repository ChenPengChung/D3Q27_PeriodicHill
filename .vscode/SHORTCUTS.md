# 快捷鍵指南 - D3Q27 PeriodicHill 專案

## MobaXterm 同步命令 (Git-like)

在 PowerShell 終端機中使用，指令與 Git 完全對應：

### Git-like 核心指令

| Git 指令 | mobaxterm 指令 | 功能說明 |
|----------|----------------|----------|
| `git status` | `mobaxterm status` | 顯示同步狀態總覽 |
| `git add` | `mobaxterm add` | 顯示待推送的變更 |
| `git diff` | `mobaxterm diff` | 比較本地與遠端差異 |
| `git push` | `mobaxterm push` | **完整推送**：上傳 + 刪除遠端多餘檔案 |
| `git pull` | `mobaxterm pull` | 從遠端下載（無刪除功能，安全模式） |
| `git fetch` | `mobaxterm fetch` | **同步下載**：下載 + 刪除本地多餘檔案 |
| `git log` | `mobaxterm log` | 查看遠端 log 檔案 |
| `git reset --hard` | `mobaxterm reset` | 只刪除遠端多餘檔案（不上傳） |
| `git clone` | `mobaxterm clone` | 從遠端完整複製到本地 |

### Pull/Clone 指定伺服器

| 命令 | 功能 |
|------|------|
| `mobaxterm pull` | 從 .87 拉取（預設） |
| `mobaxterm pull .87` | 明確指定從 .87 拉取 |
| `mobaxterm pull .154` | 從 .154 拉取 |
| `mobaxterm pull87` | 快捷：等同 `pull .87` |
| `mobaxterm pull154` | 快捷：等同 `pull .154` |
| `mobaxterm clone .154` | 從 .154 完整複製 |
| `mobaxterm log .154` | 查看 .154 的 log 檔案 |

### 同步命令完整說明（重要！）

**三大系列命令的差異：**

| 系列 | 下載/上傳 | 是否刪除 | 使用時機 | 安全性 |
|------|-----------|----------|----------|--------|
| **PULL 系列** | ⬇️ 下載 | ❌ 不刪除本地檔案 | 安全下載遠端檔案 | ✅ 最安全 |
| **FETCH 系列** | ⬇️ 下載 | ⚠️ 刪除本地多餘檔案 | 本地完全同步至遠端 | ⚠️ 會刪除 |
| **PUSH 系列** | ⬆️ 上傳 | ⚠️ 刪除遠端多餘檔案 | 遠端完全同步至本地 | ⚠️ 會刪除 |

#### PULL 系列（安全下載）

| 命令 | 功能 | 說明 |
|------|------|------|
| `mobaxterm pull` | 下載檔案（不刪除） | 從 .87 下載，本地檔案保留 |
| `mobaxterm pull .87` | 從 .87 下載 | 明確指定伺服器 |
| `mobaxterm pull .154` | 從 .154 下載 | 明確指定伺服器 |
| `mobaxterm pull87` / `pull154` | 快捷方式 | 同上 |
| `mobaxterm autopull` | 自動拉取 | 有變更才下載（不刪除） |
| `mobaxterm watchpull` | 背景監控下載 | 持續監控，自動下載新檔案 |

**特點：**
- ✅ 只下載遠端有的檔案
- ✅ 本地檔案不會被刪除（即使遠端沒有）
- ✅ 最安全的下載方式

#### FETCH 系列（完整同步）

| 命令 | 功能 | 說明 |
|------|------|------|
| `mobaxterm fetch` | 同步本地至遠端 | 下載 + **刪除本地多餘檔案** |
| `mobaxterm fetch .87` | 從 .87 完整同步 | 明確指定伺服器 |
| `mobaxterm fetch .154` | 從 .154 完整同步 | 明確指定伺服器 |
| `mobaxterm fetch87` / `fetch154` | 快捷方式 | 同上 |
| `mobaxterm autofetch` | 自動同步 | 有變更才執行（含刪除） |
| `mobaxterm watchfetch` | 背景完整同步 | 持續監控並同步 |

**特點：**
- ⬇️ 下載遠端有的檔案
- ⚠️ **刪除本地有但遠端沒有的檔案**
- 🎯 讓本地完全等於遠端狀態
- ⚠️ 使用前請確認本地沒有重要未上傳檔案

#### PUSH 系列（上傳同步）

| 命令 | 功能 | 說明 |
|------|------|------|
| `mobaxterm push` | 完整推送 | 上傳 + **刪除遠端多餘檔案** |
| `mobaxterm autopush` | 自動推送 | 有變更才執行（含刪除） |
| `mobaxterm watchpush` | 背景自動上傳 | 持續監控並上傳 |

**特點：**
- ⬆️ 上傳本地有的檔案
- ⚠️ **刪除遠端有但本地沒有的檔案**
- 🎯 讓遠端完全等於本地狀態

### Fetch 指定伺服器

| 命令 | 功能 |
|------|------|
| `mobaxterm fetch` | 從 .87 同步（預設） |
| `mobaxterm fetch .87` | 明確指定從 .87 同步 |
| `mobaxterm fetch .154` | 從 .154 同步 |
| `mobaxterm fetch87` | 快捷：等同 `fetch .87` |
| `mobaxterm fetch154` | 快捷：等同 `fetch .154` |

### 額外功能（超越 Git）

| 命令 | 功能 | 說明 |
|------|------|------|
| `mobaxterm sync` | 互動式同步 | diff → 確認 → push |
| `mobaxterm fullsync` | 完整同步 | push + reset（讓遠端完全等於本地） |
| `mobaxterm issynced` | 快速確認 | 一行顯示：`.87: [OK] \| .154: [OK]` |
| `mobaxterm watch` | **自動推送** | 監控本地檔案，變更後自動推送 (Ctrl+C 停止) |
| `mobaxterm autopush` | 有變更才推送 | 無變更時不執行 |
| `mobaxterm autopull` | 有變更才拉取 | 指定伺服器：`autopull .154` |

### 背景自動上傳 (watchpush)

| 命令 | 功能 | 說明 |
|------|------|------|
| `mobaxterm watchpush` | **啟動背景上傳** | 監控本地檔案，自動上傳到兩台伺服器 |
| `mobaxterm watchpush 5` | 自訂間隔 | 每 5 秒檢查一次（預設 10 秒） |
| `mobaxterm watchpush status` | 檢查上傳狀態 | 顯示是否執行中 + 最近上傳記錄 |
| `mobaxterm watchpush log` | 查看上傳日誌 | 完整上傳歷史 |
| `mobaxterm watchpush stop` | 停止上傳 | 關閉背景程序 |
| `mobaxterm watchpush clear` | 清除日誌 | 刪除舊日誌檔 |

**watchpush 技術細節：**
- 監控間隔：10 秒（可自訂）
- 上傳檔案：程式碼檔案 (`.h`, `.cu`, `.c`, `.cpp` 等)
- **排除項目**：`*.dat`, `log*`, `*.plt`, `a.out`, `result/`, `backup/`
- 比對方式：MD5 hash（只上傳內容有變更的檔案）
- 日誌位置：`.vscode/watchpush.log`

> **安全設計**：程式輸出檔案不會被上傳，避免覆蓋遠端正在寫入的資料

### 背景自動下載 (watchpull)

| 命令 | 功能 | 說明 |
|------|------|------|
| `mobaxterm watchpull` | **啟動背景下載** | 監控兩台伺服器，自動下載新檔案 |
| `mobaxterm watchpull .87` | 只監控 .87 | 背景執行 |
| `mobaxterm watchpull .154` | 只監控 .154 | 背景執行 |
| `mobaxterm watchpull status` | 檢查下載狀態 | 顯示是否執行中 + 最近下載記錄 |
| `mobaxterm watchpull log` | 查看下載日誌 | 完整下載歷史 |
| `mobaxterm watchpull stop` | 停止下載 | 關閉背景程序 |
| `mobaxterm watchpull clear` | 清除日誌 | 刪除舊日誌檔 |

**watchpull 技術細節：**
- 監控間隔：30 秒
- 下載檔案類型：`*.dat`, `log*`, `*.plt`, `*.vtk`, `*.bin`（**只下載輸出檔**）
- **不會下載**：程式碼檔案 (`.h`, `.cu`, `.c` 等)
- **不會刪除**：本地檔案（即使遠端沒有）
- 比對方式：MD5 hash（只下載內容有變更的檔案）
- 日誌位置：`.vscode/watchpull.log`

> **安全設計**：只拉取程式輸出，不會覆蓋你正在編輯的程式碼，也不會刪除本地檔案

### 背景完整同步 (watchfetch)

| 命令 | 功能 | 說明 |
|------|------|------|
| `mobaxterm watchfetch` | **啟動背景完整同步** | 下載 + 刪除本地多餘檔案 |
| `mobaxterm watchfetch .87` | 只監控 .87 | 背景執行 |
| `mobaxterm watchfetch .154` | 只監控 .154 | 背景執行 |
| `mobaxterm watchfetch status` | 檢查同步狀態 | 顯示是否執行中 + 最近活動 |
| `mobaxterm watchfetch log` | 查看同步日誌 | 完整同步歷史 |
| `mobaxterm watchfetch stop` | 停止同步 | 關閉背景程序 |

**watchfetch 技術細節：**
- 監控間隔：30 秒
- 下載檔案類型：`*.dat`, `log*`, `*.plt`, `*.vtk`, `*.bin`
- **會刪除**：本地有但遠端沒有的檔案
- 比對方式：MD5 hash
- 日誌位置：`.vscode/watchfetch.log`

> **⚠️ 注意**：watchfetch 會刪除本地檔案以保持與遠端同步，使用前請確認本地沒有重要未上傳檔案

### VTK 檔案自動重命名

| 命令 | 功能 | 說明 |
|------|------|------|
| `mobaxterm vtkrename` | **啟動 VTK 重命名器** | 自動將 VTK 檔案重命名為 zero-padding 格式 |
| `mobaxterm vtkrename status` | 檢查重命名器狀態 | 顯示執行狀態 + 最近重命名記錄 |
| `mobaxterm vtkrename log` | 查看重命名日誌 | 完整重命名歷史 |
| `mobaxterm vtkrename stop` | 停止重命名器 | 關閉背景程序 |

**vtkrename 技術細節：**
- 監控目錄：`result/`
- 檢查間隔：5 秒
- 重命名格式：`velocity_merged_000001.vtk`（6 位數 zero-padding）
- 最大支援：500000 步
- 日誌位置：`.vscode/vtk-renamer.log`

**重命名範例：**
```
velocity_merged_1001.vtk    → velocity_merged_001001.vtk
velocity_merged_31001.vtk   → velocity_merged_031001.vtk
velocity_merged_123456.vtk  → 保持不變（已是 6 位數）
```

> **用途**：確保 ParaView 能正確按時間步排序 VTK 檔案

### 合併狀態監控

| 命令 | 功能 |
|------|------|
| `mobaxterm bgstatus` | **一次查看所有背景程序** (watchpush/watchpull/watchfetch/vtkrename) |
| `mobaxterm syncstatus` | 查看上傳+下載狀態 (watchpush/watchpull) |

### 別名對照

| 別名 | 等同於 |
|------|--------|
| `mobaxterm check` | `mobaxterm diff` |
| `mobaxterm delete` | `mobaxterm reset` |

---

## 快速使用範例

### 基本工作流程（像 Git 一樣）

```powershell
mobaxterm status     # 查看狀態（像 git status）
mobaxterm diff       # 看差異（像 git diff）
mobaxterm push       # 推送（像 git push）
```

### 從遠端同步

```powershell
mobaxterm fetch      # 只檢查遠端（像 git fetch）
mobaxterm pull       # 拉取 .87（像 git pull）
mobaxterm pull .154  # 拉取 .154
mobaxterm clone      # 完整複製（像 git clone）
```

### 查看遠端狀態

```powershell
mobaxterm log        # 查看遠端 log 檔案
mobaxterm log .154   # 查看 .154 的 log
```

### 清理遠端

```powershell
mobaxterm reset      # 刪除本地沒有的遠端檔案
mobaxterm fullsync   # push + reset（完整同步）
```

### 自動同步模式

```powershell
# 前景自動推送（需保持終端開啟）
mobaxterm watch      # Ctrl+C 停止

# 背景自動上傳（檔案變更後自動上傳）
mobaxterm watchpush         # 啟動
mobaxterm watchpush status  # 查看狀態
mobaxterm watchpush stop    # 停止

# 背景自動下載（遠端生成檔案後自動下載）
mobaxterm watchpull         # 啟動（監控兩台）
mobaxterm watchpull .87     # 只監控 .87
mobaxterm watchpull status  # 查看狀態
mobaxterm watchpull stop    # 停止

# 所有背景程序狀態（全功能）
mobaxterm bgstatus          # 查看所有背景服務（push/pull/fetch/vtkrename）

# 合併狀態（僅上傳+下載）
mobaxterm syncstatus        # 僅查看 watchpush + watchpull
```

---

## 編譯 + 執行快捷鍵

在已連線的 SSH 終端中使用：

| 快捷鍵 | 功能 |
|--------|------|
| `Ctrl+Alt+4` | 編譯 + 用 **4 顆 GPU** 執行 |
| `Ctrl+Alt+8` | 編譯 + 用 **8 顆 GPU** 執行 |

命令內容：
```bash
cd /home/chenpengchung/D3Q27_PeriodicHill && \
nvcc main.cu -arch=sm_35 -I/home/chenpengchung/openmpi-3.0.3/include \
-L/home/chenpengchung/openmpi-3.0.3/lib -lmpi -o a.out && \
nohup mpirun -np 4 ./a.out > log$(date +%Y%m%d) 2>&1 &
```

---

## 🔑 VS Code 快捷鍵（Mac / Windows 通用）

> Mac 鍵盤：`Ctrl` = `⌃ Control`，`Alt` = `⌥ Option`
> Windows 鍵盤：`Ctrl` = `Ctrl`，`Alt` = `Alt`

| 快捷鍵 | 功能 | 說明 |
|--------|------|------|
| `Ctrl+Alt+F` | **切換子機** (Switch Node) | 即時 GPU 狀態選單 → 選擇 → 連接 |
| `Ctrl+Alt+G` | **重新連線** (Reconnect) | 即時 GPU 狀態選單 → 選擇 → 重連 |
| `Ctrl+Shift+B` | **編譯 + 執行** | 編譯 CUDA + mpirun 執行 |

---

## 🖥️ Mac vs Windows 指令完整對照

### 腳本對照

| 用途 | Mac (bash) | Windows (PowerShell) |
|------|-----------|---------------------|
| 同步腳本 | `cfdlab-mac.sh <cmd>` | `mobaxterm <cmd>` |
| SSH 腳本 | `cfdlab-mac.sh ssh/issh` | `ssh-connect.ps1` |

### SSH / GPU 相關指令

| 功能 | Mac 指令 | Windows 指令 | 狀態 |
|------|---------|-------------|------|
| SSH 連線 | `cfdlab-mac.sh ssh 87:3` | `ssh-connect.ps1 -ServerCombo "87:3"` | ✅ 雙平台 |
| **互動 GPU 選單** | `cfdlab-mac.sh issh` | `ssh-connect.ps1 -Interactive` | ✅ 雙平台 |
| GPU 總覽（精簡） | `cfdlab-mac.sh gpus` | ❌ 無 | ⚠️ Mac only |
| GPU 詳細 (nvidia-smi) | `cfdlab-mac.sh gpu 89` | ❌ 無 | ⚠️ Mac only |
| 編譯+執行 | `cfdlab-mac.sh run 87:3 4` | Task: Compile + Run | ✅ 雙平台 |
| 查看執行中工作 | `cfdlab-mac.sh jobs 87:3` | Task: Check Running Jobs | ✅ 雙平台 |
| 終止工作 | `cfdlab-mac.sh kill 87:3` | Task: Kill Running Job | ✅ 雙平台 |
| 環境檢查 | `cfdlab-mac.sh check` | ❌ 無 | ⚠️ Mac only |

### 同步指令（Git-like）

| 功能 | Mac 指令 | Windows 指令 | 狀態 |
|------|---------|-------------|------|
| **查看差異** | `cfdlab-mac.sh diff` | `mobaxterm diff` | ✅ 雙平台 |
| 查看差異（別名）| `cfdlab-mac.sh check` | `mobaxterm check` | ⚠️ 不同！Mac=SSH檢查 Win=diff |
| 查看狀態 | `cfdlab-mac.sh status` | `mobaxterm status` | ✅ 雙平台 |
| 顯示待推送 | `cfdlab-mac.sh add` | `mobaxterm add` | ✅ 雙平台 |
| **推送** | `cfdlab-mac.sh push` | `mobaxterm push` | ✅ 雙平台 |
| **拉取** | `cfdlab-mac.sh pull` | `mobaxterm pull` | ✅ 雙平台 |
| **同步下載** | `cfdlab-mac.sh fetch` | `mobaxterm fetch` | ✅ 雙平台 |
| 查看 log | `cfdlab-mac.sh log` | `mobaxterm log` | ✅ 雙平台 |
| 清理遠端 | `cfdlab-mac.sh reset` | `mobaxterm reset` | ✅ 雙平台 |
| 完整複製 | `cfdlab-mac.sh clone` | `mobaxterm clone` | ✅ 雙平台 |
| 互動式同步 | `cfdlab-mac.sh sync` | `mobaxterm sync` | ✅ 雙平台 |
| 完整同步 | `cfdlab-mac.sh fullsync` | `mobaxterm fullsync` | ✅ 雙平台 |
| 快速確認同步 | `cfdlab-mac.sh issynced` | `mobaxterm issynced` | ✅ 雙平台 |

### 指定伺服器快捷指令

| 功能 | Mac 指令 | Windows 指令 | 狀態 |
|------|---------|-------------|------|
| `pull .87` | `cfdlab-mac.sh pull87` | `mobaxterm pull87` | ✅ 雙平台 |
| `pull .89` | `cfdlab-mac.sh pull89` | ❌ 無 | ⚠️ Mac only |
| `pull .154` | `cfdlab-mac.sh pull154` | `mobaxterm pull154` | ✅ 雙平台 |
| `fetch .87` | `cfdlab-mac.sh fetch87` | `mobaxterm fetch87` | ✅ 雙平台 |
| `fetch .89` | `cfdlab-mac.sh fetch89` | ❌ 無 | ⚠️ Mac only |
| `fetch .154` | `cfdlab-mac.sh fetch154` | `mobaxterm fetch154` | ✅ 雙平台 |
| `push .87` | `cfdlab-mac.sh push87` | ❌ 無 | ⚠️ Mac only |
| `push .89` | `cfdlab-mac.sh push89` | ❌ 無 | ⚠️ Mac only |
| `push .154` | `cfdlab-mac.sh push154` | ❌ 無 | ⚠️ Mac only |
| `push all` | `cfdlab-mac.sh pushall` | ❌ 無 | ⚠️ Mac only |
| `diff .87` | `cfdlab-mac.sh diff87` | ❌ 無 | ⚠️ Mac only |
| `diff .89` | `cfdlab-mac.sh diff89` | ❌ 無 | ⚠️ Mac only |
| `diff .154` | `cfdlab-mac.sh diff154` | ❌ 無 | ⚠️ Mac only |
| `diff all` | `cfdlab-mac.sh diffall` | ❌ 無 | ⚠️ Mac only |
| `log .87` | `cfdlab-mac.sh log87` | ❌ 無 | ⚠️ Mac only |
| `log .89` | `cfdlab-mac.sh log89` | ❌ 無 | ⚠️ Mac only |
| `log .154` | `cfdlab-mac.sh log154` | ❌ 無 | ⚠️ Mac only |

### 自動同步指令

| 功能 | Mac 指令 | Windows 指令 | 狀態 |
|------|---------|-------------|------|
| 自動推送 | `cfdlab-mac.sh autopush` | `mobaxterm autopush` | ✅ 雙平台 |
| 自動拉取 | `cfdlab-mac.sh autopull` | `mobaxterm autopull` | ✅ 雙平台 |
| 自動同步 | `cfdlab-mac.sh autofetch` | `mobaxterm autofetch` | ✅ 雙平台 |
| `autopush .87` | `cfdlab-mac.sh autopush87` | ❌ 無 | ⚠️ Mac only |
| `autopush .89` | `cfdlab-mac.sh autopush89` | ❌ 無 | ⚠️ Mac only |
| `autopush .154` | `cfdlab-mac.sh autopush154` | ❌ 無 | ⚠️ Mac only |
| `autopush all` | `cfdlab-mac.sh autopushall` | ❌ 無 | ⚠️ Mac only |
| `autopull .87` | `cfdlab-mac.sh autopull87` | ❌ 無 | ⚠️ Mac only |
| `autopull .89` | `cfdlab-mac.sh autopull89` | ❌ 無 | ⚠️ Mac only |
| `autopull .154` | `cfdlab-mac.sh autopull154` | ❌ 無 | ⚠️ Mac only |
| `autofetch .87` | `cfdlab-mac.sh autofetch87` | ❌ 無 | ⚠️ Mac only |
| `autofetch .89` | `cfdlab-mac.sh autofetch89` | ❌ 無 | ⚠️ Mac only |
| `autofetch .154` | `cfdlab-mac.sh autofetch154` | ❌ 無 | ⚠️ Mac only |

### 背景監控指令

| 功能 | Mac 指令 | Windows 指令 | 狀態 |
|------|---------|-------------|------|
| 背景推送 | `cfdlab-mac.sh watchpush` | `mobaxterm watchpush` | ✅ 雙平台 |
| 背景拉取 | `cfdlab-mac.sh watchpull` | `mobaxterm watchpull` | ✅ 雙平台 |
| 背景同步 | `cfdlab-mac.sh watchfetch` | `mobaxterm watchfetch` | ✅ 雙平台 |
| VTK 重命名 | `cfdlab-mac.sh vtkrename` | `mobaxterm vtkrename` | ✅ 雙平台 |
| 全部背景狀態 | `cfdlab-mac.sh bgstatus` | `mobaxterm bgstatus` | ✅ 雙平台 |
| 同步狀態 | `cfdlab-mac.sh syncstatus` | `mobaxterm syncstatus` | ✅ 雙平台 |

> 各 watch 指令均支援子命令：`status` / `log` / `stop` / `clear`

---

## 📊 平台差異總結

### ⚠️ `check` 指令衝突

| 平台 | `check` 做什麼 |
|------|---------------|
| **Mac** | SSH 環境檢查（測試 ssh/rsync 連線） |
| **Windows** | = `diff`（比較本地 vs 遠端） |

> 以 Mac 為主：`check` = 環境檢查，`diff` = 比較差異

### ❌ Windows 缺少的功能

| 分類 | 缺少的指令 |
|------|-----------|
| **GPU** | `gpus`（GPU 總覽表）、`gpu 89`（nvidia-smi 完整輸出） |
| **.89 伺服器** | 所有 `*89` 快捷指令（pull89, fetch89, push89, diff89, log89, autopull89...） |
| **指定伺服器** | push87, push154, pushall, diff87, diff154, diffall, log87, log154 |
| **auto 快捷** | autopush87/89/154/all, autopull87/89/154, autofetch87/89/154 |
| **SSH 指令** | `ssh 87:3`, `run 87:3 4`, `jobs 87:3`, `kill 87:3` |

---

## 任務執行 (Terminal > Run Task)

| 任務名稱 | 功能 | 觸發方式 | Mac | Win |
|----------|------|----------|-----|-----|
| SSH to cfdlab | 開啟資料夾時自動連接 | 自動執行 | ✅ | ✅ |
| **Switch Node** | 切換子機（含 GPU 即時狀態） | **Ctrl+Alt+F** | ✅ | ✅ |
| **Reconnect** | 重新連線（含 GPU 即時狀態） | **Ctrl+Alt+G** | ✅ | ✅ |
| Compile + Run | 編譯並執行程式 | Ctrl+Shift+B | ✅ | ✅ |
| Check Running Jobs | 檢查執行中的作業 | 手動執行 | ✅ | ✅ |
| Kill Running Job | 終止執行中的作業 | 手動執行 | ✅ | ✅ |
| GPU Status (All) | 所有伺服器 GPU 狀態 | 手動執行 | ✅ | ❌ |
| GPU Detail (.89/.87/.154) | nvidia-smi 完整輸出 | 手動執行 | ✅ | ❌ |
| Auto Sync (Watch) | 自動推送（前景） | 手動執行 | ✅ | ✅ |
| Quick Sync | 有變更才推送 | 手動執行 | ✅ | ✅ |
| Sync Status | 查看上傳+下載狀態 | 手動執行 | ✅ | ✅ |
| Auto Upload (Start/Status/Stop) | 背景上傳管理 | 手動執行 | ✅ | ✅ |
| Auto Download (Start/Status/Stop) | 背景下載管理 | 手動執行 | ✅ | ✅ |
| Auto Download (.87/.154/.89 only) | 指定伺服器下載 | 手動執行 | ✅ | ✅ |

---

## 連線資訊

| 母機 | IP | 可連子機 | GPU 型號 |
|------|-----|----------|---------|
| .89 | 140.114.58.89 | 直連（無子機） | 8× V100-SXM2-32GB |
| .87 | 140.114.58.87 | ib2, ib3, ib5 | 8× P100-PCIE-16GB |
| .87 | 140.114.58.87 | ib6 | 8× V100-SXM2-16GB |
| .154 | 140.114.58.154 | ib4, ib9 | 8× P100-PCIE-16GB |
| .154 | 140.114.58.154 | ~~ib1, ib7~~ | ❌ 待修 |

- **工作目錄**: `/home/chenpengchung/D3Q27_PeriodicHill`
- **密碼**: `1256`

---

## 同步排除項目

以下檔案/資料夾**不會**被 push/pull 同步：

| 排除項 | 原因 |
|--------|------|
| `.git/*` | Git 版本控制（各機獨立） |
| `.vscode/*` | VS Code 設定（本地專用） |
| `backup/`, `result/`, `statistics/` | 資料夾排除 |
| `a.out`, `*.o`, `*.exe` | 編譯產物 |
| `*.dat`, `*.DAT`, `*.plt`, `*.bin`, `*.vtk` | 程式輸出檔（僅由 pull/fetch 下載） |
| `log*` | 日誌檔案（僅由 pull/fetch 下載） |

---
*最後更新: 2026-02-14*
