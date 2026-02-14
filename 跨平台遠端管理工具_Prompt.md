# Prompt：跨平台遠端伺服器管理工具（Mac / Windows 統一介面）

## 一、核心目標

開發一個跨平台（macOS + Windows）的遠端伺服器管理工具，具備以下三大功能模組：

1. **跨平台一致性**：Mac 與 Windows 的功能、指令名稱、快捷鍵行為、輸出格式完全統一。
2. **Git 風格的檔案傳輸指令**：將 MobaXterm 風格的豐富指令（pull / watchpull / autopull / push / fetch 等）與 Git CLI 的輸出格式對齊。
3. **快捷鍵選單 + GPU 監控面板**：透過快捷鍵呼出伺服器選單，同時在下方終端即時顯示各台伺服器的 GPU 使用狀況。

---

## 二、模組一：跨平台一致性規範

### 2.1 原則

- Mac 與 Windows 上的**指令名稱、參數格式、輸出訊息**必須完全相同。
- 差異僅限於底層實現（如 SSH backend），對使用者層面不可見。
- 快捷鍵採用等價映射：
  | 功能           | Windows            | macOS              |
  |----------------|--------------------|--------------------|
  | 開啟 GPU 選單  | `Ctrl + Alt + G`   | `Ctrl + Option + G`|
  | 開啟檔案傳輸   | `Ctrl + Alt + F`   | `Ctrl + Option + F`|

### 2.2 設定檔

- 使用統一的 YAML/JSON 設定檔格式，定義：
  - 伺服器清單（IP/hostname、別名、SSH port、認證方式）
  - 指令別名（alias mapping）
  - 顯示偏好（顏色主題、輸出格式）
- 設定檔路徑：
  - Windows: `%APPDATA%/remote-manager/config.yaml`
  - macOS: `~/.config/remote-manager/config.yaml`

---

## 三、模組二：Git 風格檔案傳輸指令

### 3.1 指令對照表

MobaXterm 提供的指令較為豐富（如 `pull` 有 `pull`、`watchpull`、`autopull` 等變體），但**檔案傳輸過程中的輸出顯示**應與 Git CLI 風格對齊。

| 本工具指令       | 對應 MobaXterm 功能 | 對應 Git 概念   | 說明                                      |
|------------------|---------------------|-----------------|-------------------------------------------|
| `pull`           | pull                | `git pull`      | 從遠端拉取檔案，顯示 diff-style 變更摘要  |
| `watchpull`      | watchpull           | （無直接對應）  | 監控遠端變更並自動拉取，類似 `git fetch --watch` 概念 |
| `autopull`       | autopull            | （無直接對應）  | 定時自動同步，輸出格式仍遵循 Git style     |
| `push`           | push                | `git push`      | 將本地檔案推送至遠端                      |
| `fetch`          | fetch               | `git fetch`     | 僅檢查遠端狀態，不實際傳輸                |

### 3.2 輸出格式規範（Git-aligned）

所有檔案傳輸指令的輸出必須模仿 Git 的風格，例如：

```
$ remote pull .87:/home/user/project/
remote: Checking for changes...
remote: Compressing objects: 100% (12/12), done.
Receiving files:
  modified:   src/solver.cu        (2.3 KB)
  new file:   data/mesh_v2.dat     (45.1 MB)
  deleted:    tmp/debug.log
3 files changed, 1 insertion(+), 1 deletion(-)
Transfer complete. [00:03:12]
```

### 3.3 依序多機執行 + 顏色標示

當指令需要對多台伺服器依序執行時（例如 `.87 → .89 → .154`），必須：

1. **依序執行**，不可並行（避免衝突）。
2. **每台伺服器使用不同顏色標示**，清楚區分輸出來源。
3. **每台之間有明確分隔線**。

#### 顏色配置（可自訂，預設如下）：

| 伺服器   | 顏色         | ANSI Code        |
|----------|-------------|------------------|
| `.87`    | 🟢 綠色     | `\033[32m`       |
| `.89`    | 🔵 藍色     | `\033[34m`       |
| `.154`   | 🟡 黃色     | `\033[33m`       |

#### 輸出範例：

```
══════════════════════════════════════════════════
 [1/3] 🟢 Pulling from .87 (192.168.x.87)
══════════════════════════════════════════════════
remote: Checking for changes...
  modified:   src/main.cu          (1.2 KB)
1 file changed
Transfer complete. [00:00:45] ✔

══════════════════════════════════════════════════
 [2/3] 🔵 Pulling from .89 (192.168.x.89)
══════════════════════════════════════════════════
remote: Checking for changes...
  Already up to date.
Transfer complete. [00:00:03] ✔

══════════════════════════════════════════════════
 [3/3] 🟡 Pulling from .154 (192.168.x.154)
══════════════════════════════════════════════════
remote: Checking for changes...
  modified:   config/params.yaml   (0.8 KB)
  new file:   results/output.h5    (120.5 MB)
2 files changed, 2 insertions(+)
Transfer complete. [00:01:22] ✔

══════════════════════════════════════════════════
 Summary: 3/3 servers completed successfully ✔
══════════════════════════════════════════════════
```

#### 失敗處理：

- 若某台伺服器失敗，以 🔴 紅色標示錯誤，並詢問是否繼續執行下一台。
- 提供 `--continue-on-error` flag 可跳過失敗自動繼續。

---

## 四、模組三：快捷鍵選單 + GPU 監控面板

### 4.1 快捷鍵觸發

- `Ctrl + Alt + G`（Windows）/ `Ctrl + Option + G`（macOS）：開啟 **GPU 監控選單**
- `Ctrl + Alt + F`（Windows）/ `Ctrl + Option + F`（macOS）：開啟 **檔案傳輸選單**

### 4.2 介面佈局

按下快捷鍵後，畫面分為**上下兩個區域**：

```
┌─────────────────────────────────────────────────────────┐
│  ▼ 伺服器選單（上方）                                     │
│                                                         │
│  選擇要連線的伺服器：                                     │
│  ┌─────────────────────────────────────────────────────┐│
│  │  [1] 🟢 .87  (192.168.x.87)   GPU: 2x A100        ││
│  │  [2] 🔵 .89  (192.168.x.89)   GPU: 4x RTX 3090    ││
│  │  [3] 🟡 .154 (192.168.x.154)  GPU: 2x V100        ││
│  │  [A] 全部伺服器（依序執行）                           ││
│  └─────────────────────────────────────────────────────┘│
│                                                         │
├─────────────────────────────────────────────────────────┤
│  ▼ GPU 即時監控（下方終端）                               │
│                                                         │
│  🟢 .87 — GPU Usage                                    │
│  ├─ GPU 0 [A100]: ████████░░ 78%  | Mem: 32.1/40.0 GB │
│  └─ GPU 1 [A100]: ██░░░░░░░░ 15%  | Mem:  6.2/40.0 GB │
│                                                         │
│  🔵 .89 — GPU Usage                                    │
│  ├─ GPU 0 [3090]: ██████████ 99%  | Mem: 23.8/24.0 GB │
│  ├─ GPU 1 [3090]: ██████████ 95%  | Mem: 22.1/24.0 GB │
│  ├─ GPU 2 [3090]: ░░░░░░░░░░  0%  | Mem:  0.3/24.0 GB │
│  └─ GPU 3 [3090]: ░░░░░░░░░░  2%  | Mem:  0.5/24.0 GB │
│                                                         │
│  🟡 .154 — GPU Usage                                   │
│  ├─ GPU 0 [V100]: ██████░░░░ 56%  | Mem: 18.0/32.0 GB │
│  └─ GPU 1 [V100]: ████░░░░░░ 34%  | Mem: 11.2/32.0 GB │
│                                                         │
│  🔄 Auto-refresh every 5s | Last updated: 14:32:05     │
└─────────────────────────────────────────────────────────┘
```

### 4.3 GPU 監控細節

- 透過 SSH 在各台伺服器上執行 `nvidia-smi --query-gpu=...` 取得資料。
- 顯示內容包含：
  - GPU 編號與型號
  - 使用率（百分比 + 進度條）
  - 記憶體使用量 / 總量
  - 溫度（可選）
  - 當前佔用 GPU 的 process owner（可選）
- 自動刷新間隔：預設 5 秒，可自訂。
- 使用率超過閾值時（如 >90%），以紅色高亮警示。

### 4.4 選單互動

- 使用者在上方選單中選擇伺服器編號後：
  - 若是 `G` 快捷鍵觸發：直接 SSH 連線至該伺服器。
  - 若是 `F` 快捷鍵觸發：開啟對該伺服器的檔案傳輸介面。
  - 若選擇 `[A]`：對所有伺服器依序執行指令（如 pull/push）。
- 支援鍵盤數字鍵快速選擇，也支援滑鼠點擊。

---

## 五、技術實現建議

### 5.1 建議技術棧

| 組件             | 建議方案                                    |
|------------------|---------------------------------------------|
| 跨平台 CLI/TUI   | Python + `rich` / `textual` 或 Rust + `ratatui` |
| SSH 連線         | `paramiko`（Python）或 `ssh2`（Rust）        |
| 檔案傳輸         | `rsync` over SSH 或 `scp` / `sftp`          |
| GPU 監控         | 遠端執行 `nvidia-smi --query-gpu=...`        |
| 設定管理         | YAML（`PyYAML` / `serde_yaml`）              |
| 快捷鍵攔截       | `pynput`（Python）或系統原生 API             |

### 5.2 設定檔範例（config.yaml）

```yaml
servers:
  - alias: ".87"
    host: "192.168.x.87"
    port: 22
    user: "pengzhong"
    color: "green"
    auth: "key"  # or "password"
    key_path: "~/.ssh/id_rsa"

  - alias: ".89"
    host: "192.168.x.89"
    port: 22
    user: "pengzhong"
    color: "blue"
    auth: "key"
    key_path: "~/.ssh/id_rsa"

  - alias: ".154"
    host: "192.168.x.154"
    port: 22
    user: "pengzhong"
    color: "yellow"
    auth: "key"
    key_path: "~/.ssh/id_rsa"

execution:
  order: [".87", ".89", ".154"]  # 依序執行順序
  continue_on_error: false
  
gpu_monitor:
  refresh_interval: 5  # seconds
  alert_threshold: 90  # GPU usage % to trigger red highlight

shortcuts:
  gpu_menu:
    windows: "ctrl+alt+g"
    macos: "ctrl+option+g"
  file_transfer:
    windows: "ctrl+alt+f"
    macos: "ctrl+option+f"
```

---

## 六、驗收標準（Checklist）

- [ ] Mac 與 Windows 上所有指令名稱、參數、輸出格式完全一致
- [ ] `pull` / `push` / `fetch` 輸出模仿 Git CLI 風格（含百分比、壓縮、摘要統計）
- [ ] `watchpull` / `autopull` 等進階指令正常運作，輸出同樣遵循 Git 格式
- [ ] 多機依序執行時，每台伺服器有獨立顏色標示與分隔線
- [ ] 執行順序為 `.87 → .89 → .154`，嚴格依序不並行
- [ ] 快捷鍵正確觸發選單（Windows: Ctrl+Alt+G/F, macOS: Ctrl+Option+G/F）
- [ ] 上方顯示伺服器選單，可選擇單台或全部
- [ ] 下方終端即時顯示所有伺服器的 GPU 使用狀況
- [ ] GPU 監控包含使用率、記憶體、進度條視覺化
- [ ] GPU 使用率超過閾值時以紅色高亮
- [ ] 設定檔可自訂伺服器、顏色、刷新間隔等
- [ ] 錯誤處理完善（SSH 連線失敗、超時、認證錯誤等）
