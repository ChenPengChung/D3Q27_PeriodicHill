# 快捷鍵指南 - D3Q27 PeriodicHill 專案

## MobaXterm 同步命令 (Git-like)

在 PowerShell 終端機中使用：

| 命令 | 功能 | 說明 |
|------|------|------|
| `mobaxterm check` | 檢查差異 | 比較本地與遠端 (.87 + .154) |
| `mobaxterm add .` | 顯示變更 | 列出待推送的檔案 |
| `mobaxterm push` | 推送到遠端 | 上傳到 .87 和 .154 兩台母機 |
| `mobaxterm pull` | 從 .87 拉取 | 預設從 .87 下載到本地 |
| `mobaxterm pull .154` | 從 .154 拉取 | 指定從 .154 下載 |
| `mobaxterm pull87` | 快捷拉取 .87 | 等同 `pull .87` |
| `mobaxterm pull154` | 快捷拉取 .154 | 等同 `pull .154` |
| `mobaxterm status` | 狀態總覽 | 簡要顯示同步狀態 |
| `mobaxterm sync` | 互動式同步 | check → 確認 → push |
| `mobaxterm issynced` | 快速確認 | 輸出: `.87: [OK] synced \| .154: [OK] synced` |
| `mobaxterm watch` | **自動同步** | 監控檔案變更，自動推送 (Ctrl+C 停止) |
| `mobaxterm autopush` | 自動推送 | 有變更才推送，無變更則跳過 |

### 使用範例

```powershell
# 編輯程式碼後...
mobaxterm check      # 看看有什麼改變
mobaxterm push       # 推送到兩台母機

# 或者一步到位
mobaxterm sync       # 檢查差異，確認後推送
```

### 自動同步詳細說明

#### 方法 1: `mobaxterm watch` (持續監控)

```powershell
mobaxterm watch
```

啟動後輸出：
```
[WATCH] Auto-sync enabled - monitoring file changes...
Press Ctrl+C to stop

Watching: c:\...\D3Q27_PeriodicHill
Extensions: .cu .h .c .json .md .txt .ps1
Auto-push to: .87 and .154
```

當你修改並儲存檔案時，自動顯示並推送：
```
[18:05:32] Changed : main.cu
[自動推送到 .87 和 .154]
```

**停止**: 按 `Ctrl+C`

#### 方法 2: `mobaxterm autopush` (快速檢查推送)

```powershell
mobaxterm autopush
```

- 有變更時: `[AUTO] Changes detected, pushing...` → 自動推送
- 無變更時: `[AUTO] No changes`

#### 方法 3: VS Code 任務

1. `Ctrl+Shift+P` → 輸入 `Tasks: Run Task`
2. 選擇 **Auto Sync (Watch)** 或 **Quick Sync**

### 典型工作流程

```powershell
# 1. 開始工作前，啟動自動同步
mobaxterm watch

# 2. 編輯程式碼... (自動推送)

# 3. 確認同步狀態
mobaxterm issynced
# 輸出: .87: [OK] synced | .154: [OK] synced

# 4. 從特定母機拉取
mobaxterm pull .154   # 從 .154 拉取
mobaxterm pull87      # 從 .87 拉取
```

---

## SSH 連線相關

| 快捷鍵 | 功能 | 說明 |
|--------|------|------|
| `Ctrl+Alt+F` | 切換子機 | 選擇母機 → 選擇子機 1-9 → 連接 |
| `Ctrl+Alt+G` | 從本地重連 | 斷線後重新連接：選擇母機 → 選擇子機 → 連接 |

## 任務執行 (Terminal > Run Task)

| 任務名稱 | 功能 | 觸發方式 |
|----------|------|----------|
| SSH to cfdlab | 開啟資料夾時自動連接 | 自動執行 (選擇母機 → 子機) |
| Switch Node | 切換子機 | Ctrl+Alt+F |
| Reconnect from Local | 從本地重連 | Ctrl+Alt+G |
| Compile + Run | 編譯並執行程式 | Ctrl+Shift+B |
| Check Running Jobs | 檢查執行中的作業 | 手動執行 |
| Kill Running Job | 終止執行中的作業 | 手動執行 |
| **Auto Sync (Watch)** | 自動監控同步 | 手動執行，持續運行 |
| Quick Sync | 有變更才推送 | 手動執行 |

## 終端機操作

| 快捷鍵 | 功能 |
|--------|------|
| `Shift+Enter` | 送出命令 (ESC + Enter) |

## 連線資訊

| 母機 | IP | 可連子機 | 狀態 |
|------|-----|----------|------|
| cfdlab | 140.114.58.87 | ib2, ib3, ib5, ib6 | ✅ 免密碼 |
| 另一台 | 140.114.58.154 | ib1, ib4, ib7, ib9 | ⚠️ 部分可用 |

### 子機狀態明細

| 子機 | 母機 | 狀態 | 備註 |
|------|------|------|------|
| ib1 | .154 | ❌ 待修 | 需創建 home 目錄 |
| ib2 | .87 | ✅ 正常 | |
| ib3 | .87 | ✅ 正常 | |
| ib4 | .154 | ✅ 正常 | |
| ib5 | .87 | ✅ 正常 | |
| ib6 | .87 | ✅ 正常 | |
| ib7 | .154 | ❌ 待修 | 網路不通 |
| ib9 | .154 | ✅ 正常 | |

- **工作目錄**: /home/chenpengchung/D3Q27_PeriodicHill
- **密碼**: 1256

## 連線流程

1. **開啟資料夾時**：選擇母機 → 選擇子機 → 自動連接
2. **Ctrl+Alt+F**：選擇母機 → 選擇子機 → 連接（新終端機）
3. **Ctrl+Alt+G**：選擇母機 → 選擇子機 → 從本地重新連接

---
*最後更新: 2026-02-07*
