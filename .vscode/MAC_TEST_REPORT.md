# Mac/Windows Automation Test Report

Date: 2026-02-11
Scope: `.vscode` integration, command compatibility, regression checks.

## 1) Folder migration (`scripts` removal)

- `scripts/cfdlab-mac.sh`: removed
- `scripts/MAC_AUTOMATION.md`: removed
- `scripts/` folder: removed
- Active Mac script: `.vscode/cfdlab-mac.sh`

Result: PASS

## 2) Windows compatibility regression

Checks performed:

- PowerShell parse test for `.vscode/mobaxterm.ps1`
- Run `mobaxterm.ps1` (help output)
- Run `mobaxterm.ps1 bgstatus`
- Run `mobaxterm.ps1 syncstatus`
- Run `mobaxterm.ps1 check`
- Validate `.vscode/tasks.json` can be parsed
- Verify key Windows task labels still exist

Result: PASS

## 3) Command-name parity (Windows vs Mac)

Compared top-level Windows command names in `.vscode/mobaxterm.ps1` against
case labels in `.vscode/cfdlab-mac.sh`.

Result:

- Missing Mac command names: none
- Windows names are fully covered on Mac

## 4) Mac script functional smoke tests

Checks performed:

- `bash -n .vscode/cfdlab-mac.sh` (syntax)
- `.vscode/cfdlab-mac.sh help`
- `.vscode/cfdlab-mac.sh bgstatus`
- `.vscode/cfdlab-mac.sh syncstatus all`
- `.vscode/cfdlab-mac.sh issynced`
- `.vscode/cfdlab-mac.sh check`

Result:

- Local logic commands: PASS
- Remote auth checks (.87, .154): FAIL in this environment
  - Cause: SSH authentication denied (no key/password automation configured)

## 5) Machine reachability/auth status

From this test environment:

- `.87` (`140.114.58.87`): SSH auth FAILED
- `.154` (`140.114.58.154`): SSH auth FAILED

Action needed on Mac:

1. Use SSH key auth, or
2. Install `sshpass` and set `CFDLAB_PASSWORD`, then rerun:
   - `.vscode/cfdlab-mac.sh check`

