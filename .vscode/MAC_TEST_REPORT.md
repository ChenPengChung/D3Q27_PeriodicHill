# Mac/Windows MobaXterm Command Test Report

Date: 2026-02-14

Scope:
- Mac script: `.vscode/cfdlab-mac.sh`
- Windows script: `.vscode/mobaxterm.ps1`
- Focus: all command entry points, especially `pull`, `autopull`, `watchpull` and related aliases

## 1) Test environment

### Mac runtime
- Host OS: macOS 15.5 (arm64)
- Shell: `/bin/bash`
- Method: command matrix execution with mock `ssh/rsync/sshpass` to validate command routing and daemon lifecycle safely

### Windows runtime (PowerShell)
- Runtime: PowerShell 7.5.4 (portable `pwsh`)
- Method: execute `mobaxterm.ps1` command matrix with mock `plink/pscp/powershell.exe`
- Note: this validates Windows-script behavior under PowerShell runtime; final production verification should still be done once on a real Windows host

## 2) Overall results

| Platform | Commands tested | Pass (exit code 0) | Non-zero |
|---|---:|---:|---:|
| Mac (`cfdlab-mac.sh`) | 75 | 75 | 0 |
| Windows (`mobaxterm.ps1`) | 74 | 73 | 1 |

Windows non-zero command:
- `watch` -> timeout (expected): foreground continuous monitor by design, requires manual `Ctrl+C`

## 3) Key command verification (requested focus)

### Mac
- `pull`, `pull87`, `pull89`, `pull154`: PASS
- `autopull`, `autopull87`, `autopull89`, `autopull154`: PASS
- `watchpull` lifecycle:
  - `watchpull start`: PASS
  - `watchpull status`: PASS
  - `watchpull log`: PASS
  - `watchpull clear`: PASS
  - `watchpull stop`: PASS

### Windows
- `pull`, `pull87`, `pull89`, `pull154`: PASS
- `autopull`, `autopull87`, `autopull154`: PASS
- `watchpull` lifecycle:
  - `watchpull start`: PASS
  - `watchpull status`: PASS
  - `watchpull log`: PASS
  - `watchpull clear`: PASS
  - `watchpull stop`: PASS

## 4) Important findings

1. `mobaxterm autopull89` currently falls back to `.87` (not `.89`).
2. `mobaxterm autofetch89` currently falls back to `.87` (not `.89`).
3. `mobaxterm autopush87` / `autopush89` / `autopush154` currently push to all servers, not only the named server.
4. `mobaxterm watch` is a foreground watcher and will not return automatically.

## 5) Recommendation

- Daily use can proceed for most commands.
- Before production use on Windows, fix/verify the 3 alias-routing issues above (`autopull89`, `autofetch89`, `autopush87/89/154`).
- For background mode on Windows, prefer `watchpush/watchpull/watchfetch`; use `watch` only when you want foreground continuous monitoring.
