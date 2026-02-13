# D3Q27 Sync 快速操作手冊 (Mac + Windows)

本手冊整合：
- `.vscode/MAC_TEST_REPORT.md` (最新測試結論)
- `.vscode/MAC_VSCODE_SETUP.md` (環境與 VS Code 任務)
- 舊版 `SHORTCUTS` 指令對照

## 1) 先理解三種同步模式

| 系列 | 方向 | 刪除行為 | 典型用途 |
|---|---|---|---|
| `pull` | 遠端 -> 本地 | 不刪本地 | 安全下載結果檔 |
| `fetch` | 遠端 -> 本地 | 會刪本地多餘檔 | 讓本地完全對齊遠端 |
| `push` | 本地 -> 遠端 | 會刪遠端多餘檔 | 讓遠端完全對齊本地 |

## 2) 最常用日常流程

```bash
mobaxterm status
mobaxterm diff
mobaxterm push
mobaxterm pull 89
mobaxterm watchpull 87
```

## 3) 核心命令索引

### 3.1 Git-like 核心
- `status`, `diff`, `add`
- `push`, `pull`, `fetch`, `log`
- `reset`, `delete`, `clone`, `sync`, `fullsync`, `issynced`

### 3.2 指定伺服器快捷
- `pull87`, `pull89`, `pull154`
- `fetch87`, `fetch89`, `fetch154`
- `push87`, `push89`, `push154`, `pushall`
- `diff87`, `diff89`, `diff154`, `diffall`
- `log87`, `log89`, `log154`

### 3.3 自動同步
- `autopull`, `autofetch`, `autopush`
- `autopull87/89/154`
- `autofetch87/89/154`
- `autopush87/89/154/all`

### 3.4 背景服務
- `watchpush [interval]`
- `watchpull [server] [interval]`
- `watchfetch [server] [interval]`
- `vtkrename [interval]`

每個 background 命令都支援：
- `status`
- `log`
- `clear`
- `stop`

範例：
```bash
mobaxterm watchpull 87 30
mobaxterm watchpull status
mobaxterm watchpull stop
```

### 3.5 SSH/GPU 操作
- `ssh 87:3`
- `issh` (互動選單)
- `run 87:3 4`
- `jobs 87:3`
- `kill 87:3`
- `gpus`
- `gpu 89`

## 4) 平台差異 (依 2026-02-14 實測)

### Mac (`cfdlab-mac.sh`)
- 命令矩陣 75/75 全部通過
- `watch*` / `vtkrename` 啟停與狀態查詢正常

### Windows (`mobaxterm.ps1`)
- 命令矩陣 73/74 通過
- `watch` 會前景持續監控，需手動 `Ctrl+C`（非錯誤）
- 目前已知路由異常：
  - `autopull89` -> 實際走 `.87`
  - `autofetch89` -> 實際走 `.87`
  - `autopush87/89/154` -> 實際對全部伺服器動作

## 5) 安全使用建議

1. 不確定時，優先用 `pull` 而非 `fetch`。
2. `push` / `fetch` 可能刪檔，先跑 `diff` 或 `status`。
3. `watch` (Windows) 為前景模式；想背景執行請用 `watchpush/watchpull/watchfetch`。
4. 若要做單一伺服器自動上傳，Windows 目前先改用 `push87/push89/push154`（不要依賴 `autopush87/89/154`）。

## 6) 速查範例

```bash
# 安全抓取 .89 輸出
mobaxterm pull89

# 僅監控 .87 自動下載
mobaxterm watchpull 87
mobaxterm watchpull status

# 推送到指定伺服器
mobaxterm push154

# 查看 GPU 與工作狀態
mobaxterm gpus
mobaxterm jobs 87:3
```

---
Last updated: 2026-02-14
