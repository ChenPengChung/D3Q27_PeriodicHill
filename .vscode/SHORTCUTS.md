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
| `git pull` | `mobaxterm pull` | 從遠端拉取（預設 .87） |
| `git fetch` | `mobaxterm fetch` | 只檢查遠端狀態（不下載） |
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

### 額外功能（超越 Git）

| 命令 | 功能 | 說明 |
|------|------|------|
| `mobaxterm sync` | 互動式同步 | diff → 確認 → push |
| `mobaxterm fullsync` | 完整同步 | push + reset（讓遠端完全等於本地） |
| `mobaxterm issynced` | 快速確認 | 一行顯示：`.87: [OK] \| .154: [OK]` |
| `mobaxterm watch` | **自動同步** | 監控檔案，自動推送 (Ctrl+C 停止) |
| `mobaxterm autopush` | 有變更才推送 | 無變更時不執行 |

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
mobaxterm watch      # 啟動自動監控（編輯後自動推送）
# Ctrl+C 停止
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

## SSH 連線相關

| 快捷鍵 | 功能 | 說明 |
|--------|------|------|
| `Ctrl+Alt+F` | 切換子機 | 選擇母機 → 選擇子機 1-9 → 連接 |
| `Ctrl+Alt+G` | 從本地重連 | 斷線後重新連接 |

## 任務執行 (Terminal > Run Task)

| 任務名稱 | 功能 | 觸發方式 |
|----------|------|----------|
| SSH to cfdlab | 開啟資料夾時自動連接 | 自動執行 |
| Switch Node | 切換子機 | Ctrl+Alt+F |
| Reconnect | 從本地重連 | Ctrl+Alt+G |
| Compile + Run | 編譯並執行程式 | Ctrl+Shift+B |
| Check Running Jobs | 檢查執行中的作業 | 手動執行 |
| Kill Running Job | 終止執行中的作業 | 手動執行 |
| **Auto Sync (Watch)** | 自動監控同步 | 手動執行 |
| Quick Sync | 有變更才推送 | 手動執行 |

---

## 連線資訊

| 母機 | IP | 可連子機 |
|------|-----|----------|
| cfdlab | 140.114.58.87 | ib2, ib3, ib5, ib6 |
| 另一台 | 140.114.58.154 | ib1, ib4, ib7, ib9 |

### 子機狀態

| 子機 | 母機 | 狀態 |
|------|------|------|
| ib2, ib3, ib5, ib6 | .87 | ✅ 正常 |
| ib4, ib9 | .154 | ✅ 正常 |
| ib1, ib7 | .154 | ❌ 待修 |

- **工作目錄**: `/home/chenpengchung/D3Q27_PeriodicHill`
- **密碼**: `1256`

---
*最後更新: 2026-02-07*
