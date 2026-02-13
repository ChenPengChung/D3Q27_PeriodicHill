# VS Code Setup Guide (Mac-focused, Windows-aligned)

This project keeps command naming aligned between:
- Mac: `.vscode/cfdlab-mac.sh`
- Windows: `.vscode/mobaxterm.ps1`

## 1) Mac prerequisites

```bash
brew install rsync
brew install hudochenkov/sshpass/sshpass   # optional but recommended for password mode
```

Built-in on macOS:
- `ssh`

## 2) Mac first-time setup

```bash
chmod +x .vscode/cfdlab-mac.sh
chmod +x .vscode/setup-alias.sh
.vscode/setup-alias.sh
```

After reopening terminal, you can use:
- `mobaxterm <command>`

## 3) VS Code task usage

1. Open this workspace in VS Code.
2. `Terminal -> Run Task...`
3. Use tasks with `[Mac]` prefix, for example:
   - `[Mac] Check Environment`
   - `[Mac] Sync Status`
   - `[Mac] Auto Pull (once)`
   - `[Mac] Watch Pull`
   - `[Mac] Watch Push`

## 4) Input fields used by tasks

- `macServerCombo`: `87:3`, `154:4`, `89:0`
- `macServer`: `87`, `89`, `154`
- `macServerOrAll`: `all`, `87`, `89`, `154`
- `macGpuCount`: `4` or `8`

## 5) Command compatibility baseline

Both platforms provide the same core command names:
- `diff`, `status`, `add`, `push`, `pull`, `fetch`, `log`
- `reset`, `delete`, `clone`, `sync`, `fullsync`, `issynced`
- `autopush`, `autopull`, `autofetch`
- `watch`, `watchpush`, `watchpull`, `watchfetch`
- `syncstatus`, `bgstatus`, `vtkrename`
- `ssh`, `run`, `jobs`, `kill`, `gpus`, `gpu`

See full daily command handbook in `.vscode/SHORTCUTS.md`.

## 6) Known Windows behavior differences from latest tests

- `autopull89` / `autofetch89` currently route to `.87`.
- `autopush87/89/154` currently operate all servers.
- `watch` is foreground infinite monitor by design.
