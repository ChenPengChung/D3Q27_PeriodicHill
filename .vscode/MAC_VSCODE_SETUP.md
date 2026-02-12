# VS Code Mac Integration Guide

This repository now uses a single Mac automation script inside `.vscode`:

- Script: `.vscode/cfdlab-mac.sh`
- VS Code tasks: `.vscode/tasks.json` (labels start with `[Mac]`)

## Folder layout

Keep these files:

- `.vscode/cfdlab-mac.sh`
- `.vscode/tasks.json`
- `.vscode/MAC_VSCODE_SETUP.md`

The old `scripts/` folder is no longer required.

## macOS prerequisites

```bash
brew install rsync
```

`ssh` is built in on macOS.

Optional (if you want password-based automation like Windows):

```bash
brew install hudochenkov/sshpass/sshpass
```

## First-time setup on Mac

```bash
chmod +x .vscode/cfdlab-mac.sh
```

## Run tasks in VS Code

1. Open project in VS Code.
2. Open `Terminal -> Run Task...`.
3. Run any task with `[Mac]` prefix, for example:
   - `[Mac] Check Environment`
   - `[Mac] SSH to cfdlab`
   - `[Mac] Compile + Run`
   - `[Mac] Sync Status`
   - `[Mac] Auto Pull (once)`
   - `[Mac] Auto Push (once)`
   - `[Mac] Watch Pull`
   - `[Mac] Watch Push`

## Input fields used by tasks

- `macServerCombo`: server and node, for example `87:3`
- `macServer`: one server (`87` or `154`)
- `macServerOrAll`: `all`, `87`, or `154`
- `macGpuCount`: GPU count (`4` or `8`)

## Command naming compatibility

`.vscode/cfdlab-mac.sh` supports the same core command names as Windows `mobaxterm.ps1`:

- `diff`, `check`, `status`, `add`, `push`, `pull`, `fetch`, `log`
- `reset`, `delete`, `clone`, `sync`, `fullsync`, `issynced`
- `autopush`, `autopull`, `autofetch`
- `watch`, `watchpush`, `watchpull`, `watchfetch`
- `syncstatus`, `bgstatus`, `vtkrename`
- `pull87`, `pull154`, `fetch87`, `fetch154`

## Troubleshooting

- `Missing command: rsync` -> run `brew install rsync`
- Password mode -> install `sshpass` and set:
  `export CFDLAB_PASSWORD='your_password'`
- SSH fails -> test manually:
  `ssh chenpengchung@140.114.58.87`
- Task runs but no sync -> run `[Mac] Check Environment` and `[Mac] Sync Status`
