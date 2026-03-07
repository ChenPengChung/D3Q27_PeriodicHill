#!/usr/bin/env bash
set -euo pipefail

# Ensure Homebrew paths are available (macOS arm64 & intel)
[[ -d /opt/homebrew/bin ]] && export PATH="/opt/homebrew/bin:$PATH"
[[ -d /usr/local/bin ]]    && export PATH="/usr/local/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ========== Auto-setup 'mobaxterm' alias ==========
# This will add alias to your shell profile if not already added
function auto_setup_alias() {
  local script_path="$SCRIPT_DIR/Zsh_mainsystem.sh"
  local shell_profile=""

  # Determine shell profile
  if [[ -n "${ZSH_VERSION:-}" ]]; then
    shell_profile="$HOME/.zshrc"
  elif [[ -f "$HOME/.bash_profile" ]]; then
    shell_profile="$HOME/.bash_profile"
  elif [[ -f "$HOME/.bashrc" ]]; then
    shell_profile="$HOME/.bashrc"
  else
    shell_profile="$HOME/.profile"
  fi

  # Check if alias already exists
  if ! grep -q "alias mobaxterm=" "$shell_profile" 2>/dev/null; then
    echo "" >> "$shell_profile"
    echo "# MobaXterm alias (auto-added)" >> "$shell_profile"
    echo "alias mobaxterm='$script_path'" >> "$shell_profile"
    echo "[AUTO-SETUP] Added 'mobaxterm' alias to $shell_profile"
    echo "             Run 'source $shell_profile' or restart terminal to use."
    echo ""
  fi
}

# Run auto-setup on first use (only if running interactively)
if [[ -t 1 ]] && [[ "${1:-}" != "__daemon_loop" ]]; then
  auto_setup_alias
fi
# ========== End Auto-setup ==========
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
STATE_DIR="$WORKSPACE_DIR/.vscode"

CFDLAB_USER="${CFDLAB_USER:-chenpengchung}"
# 自動根據本地資料夾名稱生成遠端路徑
LOCAL_FOLDER_NAME="$(basename "$WORKSPACE_DIR")"
CFDLAB_REMOTE_PATH="${CFDLAB_REMOTE_PATH:-/home/chenpengchung/${LOCAL_FOLDER_NAME}}"
CFDLAB_DEFAULT_NODE="${CFDLAB_DEFAULT_NODE:-3}"
CFDLAB_DEFAULT_GPU_COUNT="${CFDLAB_DEFAULT_GPU_COUNT:-4}"
CFDLAB_NVCC_ARCH="${CFDLAB_NVCC_ARCH:-sm_35}"
CFDLAB_MPI_INCLUDE="${CFDLAB_MPI_INCLUDE:-/home/chenpengchung/openmpi-3.0.3/include}"
CFDLAB_MPI_LIB="${CFDLAB_MPI_LIB:-/home/chenpengchung/openmpi-3.0.3/lib}"
CFDLAB_ASSUME_YES="${CFDLAB_ASSUME_YES:-0}"
CFDLAB_PASSWORD="${CFDLAB_PASSWORD:-1256}"
CFDLAB_SSH_OPTS="${CFDLAB_SSH_OPTS:--o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new}"

WATCHPUSH_PID="$STATE_DIR/watchpush.pid"
WATCHPUSH_LOG="$STATE_DIR/watchpush.log"
WATCHPULL_PID="$STATE_DIR/watchpull.pid"
WATCHPULL_LOG="$STATE_DIR/watchpull.log"
WATCHFETCH_PID="$STATE_DIR/watchfetch.pid"
WATCHFETCH_LOG="$STATE_DIR/watchfetch.log"
VTKRENAME_PID="$STATE_DIR/vtk-renamer.pid"
VTKRENAME_LOG="$STATE_DIR/vtk-renamer.log"

# ========== ANSI Color Constants (Git-style output) ==========
CLR_87='\033[32m'    # green
CLR_89='\033[34m'    # blue
CLR_154='\033[33m'   # yellow
if [[ -t 1 ]]; then
  CLR_ERR='\033[31m'   # red
  CLR_OK='\033[32m'    # green (success)
  CLR_DIM='\033[90m'   # dim gray
  CLR_BOLD='\033[1m'   # bold
  CLR_RST='\033[0m'    # reset
else
  CLR_ERR='' CLR_OK='' CLR_DIM='' CLR_BOLD='' CLR_RST=''
  CLR_87='' CLR_89='' CLR_154=''
fi

function server_color() {
  case "$1" in
    87)  printf '%b' "$CLR_87" ;;
    89)  printf '%b' "$CLR_89" ;;
    154) printf '%b' "$CLR_154" ;;
    *)   printf '%b' "$CLR_RST" ;;
  esac
}

function server_emoji() {
  case "$1" in
    87)  echo "🟢" ;;
    89)  echo "🔵" ;;
    154) echo "🟡" ;;
    *)   echo "⚪" ;;
  esac
}

# ========== Git-style output formatting ==========

# Convert rsync --itemize-changes line to Git-style label
# Input: rsync itemize line, e.g. ">f+++++++++++ path/to/file"
# Output: "  new file:   path/to/file"
function format_rsync_line() {
  local line="$1"
  local flags="${line%% *}"
  local path="${line#* }"
  path="${path#./}"

  [[ -z "$path" || "$path" == "." || "$path" == "./" ]] && return

  local label=""
  case "$flags" in
    *deleting*)    label="${CLR_ERR}  deleted:    ${CLR_RST}" ;;
    '>f+++++++'+*) label="${CLR_OK}  new file:   ${CLR_RST}" ;;
    '<f+++++++'+*) label="${CLR_OK}  new file:   ${CLR_RST}" ;;
    '>f'*)         label="  modified:  " ;;
    '<f'*)         label="  modified:  " ;;
    'cd+++++++'+*) label="${CLR_OK}  new dir:    ${CLR_RST}" ;;
    '.d'*)         return ;;  # directory metadata change, skip
    *)             label="  changed:   " ;;
  esac

  printf '%b%s\n' "$label" "$path"
}

# Format rsync itemize output to Git-style summary
# stdin: rsync --itemize-changes output
# stdout: formatted git-style output
function format_rsync_output() {
  local new_count=0
  local mod_count=0
  local del_count=0
  local total=0
  local lines=()

  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    # Only process valid rsync itemize lines (11-char flags field + space + path)
    # Valid flags start with: > < c . * h (rsync itemize format)
    local flags="${line%% *}"
    if [[ ! "$flags" =~ ^[.\<\>ch\*][fdLDS] ]]; then
      # Also accept *deleting lines
      if [[ "$flags" != *deleting ]]; then
        continue  # skip non-rsync lines (e.g. SSH banners)
      fi
    fi
    # Skip directory-only metadata lines
    [[ "$flags" == .d* ]] && continue
    
    local formatted
    formatted="$(format_rsync_line "$line")"
    [[ -z "$formatted" ]] && continue

    lines+=("$formatted")
    ((total++))

    case "$flags" in
      *deleting*)    ((del_count++)) ;;
      *'++++++'*)    ((new_count++)) ;;
      *)             ((mod_count++)) ;;
    esac
  done

  if [[ "$total" -eq 0 ]]; then
    echo "  Already up to date."
    return
  fi

  for l in "${lines[@]}"; do
    printf '%s\n' "$l"
  done

  # Summary line
  local summary="${total} file(s) changed"
  [[ "$new_count" -gt 0 ]] && summary+=", ${new_count} insertion(+)"
  [[ "$del_count" -gt 0 ]] && summary+=", ${del_count} deletion(-)"
  echo "$summary"
}

# Classify an rsync itemize-changes flag string
# Returns: new / deleted / modified / skip
function classify_rsync_flags() {
  local flags="$1"
  case "$flags" in
    *deleting*)    echo "deleted" ;;
    *'++++++'*)    echo "new" ;;
    .d*)           echo "skip" ;;
    *)             echo "modified" ;;
  esac
}

# Run rsync with Git-style real-time progress output
# Args: direction(push|pull|fetch) server delete_mode
function git_style_transfer() {
  local direction="$1"
  local server="$2"
  local delete_mode="$3"
  local host
  local args=()
  local start_time
  local is_tty=0

  [[ -t 1 ]] && is_tty=1

  host="$(resolve_host "$server")"
  start_time="$(date +%s)"

  # Build rsync args
  while IFS= read -r line; do
    args+=("$line")
  done < <(build_arg_array "$direction" "$delete_mode")

  local src dst action_verb
  if [[ "$direction" == "push" ]]; then
    src="${WORKSPACE_DIR}/"
    dst="${CFDLAB_USER}@${host}:${CFDLAB_REMOTE_PATH}/"
    action_verb="Uploading"
    ensure_remote_dir "$server"
  else
    src="${CFDLAB_USER}@${host}:${CFDLAB_REMOTE_PATH}/"
    dst="${WORKSPACE_DIR}/"
    if [[ "$direction" == "fetch" ]]; then
      action_verb="Fetching"
    else
      action_verb="Downloading"
    fi
  fi

  # ── Phase 1: dry-run to count total objects ──
  if [[ "$is_tty" -eq 1 ]]; then
    printf '%bremote: Enumerating objects...%b' "$CLR_DIM" "$CLR_RST"
  else
    printf '%bremote: Enumerating objects...%b\n' "$CLR_DIM" "$CLR_RST"
  fi

  local dry_args=("${args[@]}" --dry-run --itemize-changes)
  local dry_output
  dry_output="$(rsync "${dry_args[@]}" "$src" "$dst" 2>/dev/null)" || true

  local total_files=0
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    local flags="${line%% *}"
    if [[ "$flags" =~ ^[.\<\>ch\*][fdLDS] ]] || [[ "$flags" == *deleting ]]; then
      local cls
      cls="$(classify_rsync_flags "$flags")"
      [[ "$cls" == "skip" ]] && continue
      ((total_files++))
    fi
  done <<< "$dry_output"

  if [[ "$is_tty" -eq 1 ]]; then
    printf '\r\033[K%bremote: Enumerating objects: %d, done.%b\n' "$CLR_DIM" "$total_files" "$CLR_RST"
  else
    printf '%bremote: Enumerating objects: %d, done.%b\n' "$CLR_DIM" "$total_files" "$CLR_RST"
  fi

  if [[ "$total_files" -eq 0 ]]; then
    echo "  Already up to date."
    printf 'Transfer complete. [00:00] ✔\n'
    return 0
  fi

  # ── Phase 1.5: compute per-file line diffs (for modified text files) ──
  # Use temp file for key-value storage (bash 3.2 compat, no declare -A)
  local _linestats_file
  _linestats_file="$(mktemp /tmp/linestats.XXXXXX)"
  local total_line_adds=0
  local total_line_dels=0

  if [[ "$is_tty" -eq 1 ]]; then
    printf '%bremote: Computing line changes...%b' "$CLR_DIM" "$CLR_RST"
  fi

  local diff_idx=0
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    local d_flags="${line%% *}"
    if [[ "$d_flags" =~ ^[.\<\>ch\*][fdLDS] ]] || [[ "$d_flags" == *deleting ]]; then
      local d_cls
      d_cls="$(classify_rsync_flags "$d_flags")"
      [[ "$d_cls" == "skip" ]] && continue

      local d_path="${line#* }"
      d_path="${d_path#./}"
      ((diff_idx++))

      if [[ "$is_tty" -eq 1 ]]; then
        printf '\r\033[K%bremote: Computing line changes... (%d/%d) %s%b' \
          "$CLR_DIM" "$diff_idx" "$total_files" "$d_path" "$CLR_RST"
      fi

      if [[ "$d_cls" == "modified" ]]; then
        # Only compute diff for text files
        if is_text_file "$d_path"; then
          local local_file="${WORKSPACE_DIR}/${d_path}"
          if [[ "$direction" == "push" ]]; then
            # Push: compare local (new) vs remote (old)
            local tmp_remote
            tmp_remote="$(fetch_remote_to_temp "$server" "$d_path")"
            if [[ -f "$tmp_remote" && -s "$tmp_remote" ]]; then
              local adds=0 dels=0
              while IFS= read -r dline; do
                case "$dline" in
                  +*) ((adds++)) ;;
                  -*) ((dels++)) ;;
                esac
              done < <(diff -u "$tmp_remote" "$local_file" 2>/dev/null | tail -n +3 | grep -E '^\+|^-' | grep -v '^+++\|^---')
              echo "${d_path}|+${adds}|-${dels}" >> "$_linestats_file"
              total_line_adds=$((total_line_adds + adds))
              total_line_dels=$((total_line_dels + dels))
            fi
            rm -f "$tmp_remote" 2>/dev/null
          else
            # Pull/Fetch: compare remote (new) vs local (old)
            local tmp_remote
            tmp_remote="$(fetch_remote_to_temp "$server" "$d_path")"
            if [[ -f "$tmp_remote" && -s "$tmp_remote" && -f "$local_file" ]]; then
              local adds=0 dels=0
              while IFS= read -r dline; do
                case "$dline" in
                  +*) ((adds++)) ;;
                  -*) ((dels++)) ;;
                esac
              done < <(diff -u "$local_file" "$tmp_remote" 2>/dev/null | tail -n +3 | grep -E '^\+|^-' | grep -v '^+++\|^---')
              echo "${d_path}|+${adds}|-${dels}" >> "$_linestats_file"
              total_line_adds=$((total_line_adds + adds))
              total_line_dels=$((total_line_dels + dels))
            fi
            rm -f "$tmp_remote" 2>/dev/null
          fi
        fi
      elif [[ "$d_cls" == "new" ]]; then
        # New file: count all lines as additions (for text files)
        if is_text_file "$d_path"; then
          local lines_count=0
          if [[ "$direction" == "push" ]]; then
            local local_file="${WORKSPACE_DIR}/${d_path}"
            [[ -f "$local_file" ]] && lines_count=$(wc -l < "$local_file" 2>/dev/null | tr -d ' ')
          else
            lines_count=$(ssh_batch_exec "$host" "wc -l < '${CFDLAB_REMOTE_PATH}/${d_path}'" 2>/dev/null | tr -d ' ')
          fi
          lines_count=${lines_count:-0}
          echo "${d_path}|+${lines_count}|-0" >> "$_linestats_file"
          total_line_adds=$((total_line_adds + lines_count))
        fi
      elif [[ "$d_cls" == "deleted" ]]; then
        # Deleted file: count all lines as deletions (for text files)
        if is_text_file "$d_path"; then
          local lines_count=0
          if [[ "$direction" == "push" ]]; then
            lines_count=$(ssh_batch_exec "$host" "wc -l < '${CFDLAB_REMOTE_PATH}/${d_path}'" 2>/dev/null | tr -d ' ')
          else
            local local_file="${WORKSPACE_DIR}/${d_path}"
            [[ -f "$local_file" ]] && lines_count=$(wc -l < "$local_file" 2>/dev/null | tr -d ' ')
          fi
          lines_count=${lines_count:-0}
          echo "${d_path}|+0|-${lines_count}" >> "$_linestats_file"
          total_line_dels=$((total_line_dels + lines_count))
        fi
      fi
    fi
  done <<< "$dry_output"

  if [[ "$is_tty" -eq 1 ]]; then
    printf '\r\033[K%bremote: Computing line changes: done.%b\n' "$CLR_DIM" "$CLR_RST"
  else
    printf '%bremote: Computing line changes: done.%b\n' "$CLR_DIM" "$CLR_RST"
  fi

  # ── Phase 2: real transfer with real-time progress ──
  # Retry wrapper: up to 3 attempts
  local max_retries=3
  local attempt=0
  local rsync_exit=1
  local transfer_args=("${args[@]}" --itemize-changes)
  local current=0
  local new_count=0 mod_count=0 del_count=0
  local formatted_lines=()
  local phase2_start
  phase2_start="$(date +%s)"

  while [[ $attempt -lt $max_retries ]]; do
    ((attempt++))
    if [[ $attempt -gt 1 ]]; then
      printf '%b⚠ Retry %d/%d (waiting 3s...)%b\n' "$CLR_ERR" "$attempt" "$max_retries" "$CLR_RST"
      sleep 3
    fi

    current=0; new_count=0; mod_count=0; del_count=0
    formatted_lines=()
    phase2_start="$(date +%s)"

    while IFS= read -r line; do
      [[ -z "$line" ]] && continue
      local flags="${line%% *}"
      # Only process valid rsync itemize lines
      if [[ "$flags" =~ ^[.\<\>ch\*][fdLDS] ]] || [[ "$flags" == *deleting ]]; then
        local cls
        cls="$(classify_rsync_flags "$flags")"
        [[ "$cls" == "skip" ]] && continue

        ((current++))
        local path="${line#* }"
        path="${path#./}"
        local pct=$((current * 100 / total_files))

        # Count by type
        case "$cls" in
          new)      ((new_count++)) ;;
          deleted)  ((del_count++)) ;;
          modified) ((mod_count++)) ;;
        esac

        # Calculate speed & ETA
        local now_ts elapsed_so_far speed_str eta_str bar_str
        now_ts="$(date +%s)"
        elapsed_so_far=$((now_ts - phase2_start))
        if [[ $elapsed_so_far -gt 0 && $current -gt 0 ]]; then
          local files_per_sec=$(( (current * 100) / elapsed_so_far ))  # x100 for precision
          local remaining_files=$((total_files - current))
          local eta_s=0
          if [[ $files_per_sec -gt 0 ]]; then
            eta_s=$(( (remaining_files * 100) / files_per_sec ))
          fi
          if [[ $eta_s -ge 60 ]]; then
            eta_str="$((eta_s / 60))m$((eta_s % 60))s"
          else
            eta_str="${eta_s}s"
          fi
          speed_str="$(printf '%.1f' "$(echo "scale=1; $current / $elapsed_so_far" | bc 2>/dev/null || echo "0")")"
          speed_str="${speed_str} files/s"
        else
          eta_str="--"
          speed_str="--"
        fi

        # Progress bar (20 chars wide)
        if [[ "$is_tty" -eq 1 ]]; then
          local bar_width=20
          local filled=$((pct * bar_width / 100))
          local empty=$((bar_width - filled))
          bar_str=""
          local i
          for ((i=0; i<filled; i++)); do bar_str+="━"; done
          for ((i=0; i<empty; i++)); do bar_str+="─"; done
        fi

        # Show real-time progress
        if [[ "$is_tty" -eq 1 ]]; then
          printf '\r\033[K%b%s%b %3d%% %b(%d/%d)%b %s | ⏱ %s  📁 %s' \
            "$CLR_DIM" "$bar_str" "$CLR_RST" "$pct" \
            "$CLR_DIM" "$current" "$total_files" "$CLR_RST" \
            "$speed_str" "$eta_str" "$path"
        else
          printf '[%3d%%] (%d/%d) %s\n' "$pct" "$current" "$total_files" "$path"
        fi

        # Store formatted line for summary
        local fmt
        fmt="$(format_rsync_line "$line")"
        [[ -n "$fmt" ]] && formatted_lines+=("$fmt")
      fi
    done < <(rsync "${transfer_args[@]}" "$src" "$dst" 2>&1)
    rsync_exit=${PIPESTATUS[0]:-$?}

    # If rsync succeeded or partial transfer with all files done, break
    if [[ $rsync_exit -eq 0 ]] || [[ $current -ge $total_files ]]; then
      break
    fi
    printf '\n%b⚠ rsync exited with code %d%b\n' "$CLR_ERR" "$rsync_exit" "$CLR_RST"
  done

  # ── Phase 3: summary ──
  local end_time elapsed_s mins secs
  end_time="$(date +%s)"
  elapsed_s=$((end_time - start_time))
  mins=$((elapsed_s / 60))
  secs=$((elapsed_s % 60))

  if [[ "$is_tty" -eq 1 ]]; then
    printf '\r\033[K%b%s objects: 100%% (%d/%d), done.%b\n' \
      "$CLR_DIM" "$action_verb" "$total_files" "$total_files" "$CLR_RST"
  else
    printf '%b%s objects: 100%% (%d/%d), done.%b\n' \
      "$CLR_DIM" "$action_verb" "$total_files" "$total_files" "$CLR_RST"
  fi

  # Print formatted file list with per-file line stats
  for l in "${formatted_lines[@]+${formatted_lines[@]}}"; do
    # Extract path from formatted line for line stats lookup
    local fpath=""
    # Try to extract the path (after label text)
    fpath="$(echo "$l" | sed 's/.*\(new file:\|modified:\|deleted:\|new dir:\|changed:\)[[:space:]]*//' | sed 's/\x1b\[[0-9;]*m//g' | tr -d '[:space:]' | head -1)"
    # Fallback: trim ANSI codes and get last word
    if [[ -z "$fpath" ]]; then
      fpath="$(echo "$l" | sed 's/\x1b\[[0-9;]*m//g' | awk '{print $NF}')"
    fi

    local stat_str=""
    local _ls_line=""
    if [[ -f "$_linestats_file" ]]; then
      _ls_line="$(grep -F "${fpath}|" "$_linestats_file" | head -1)"
    fi
    if [[ -n "$_ls_line" ]]; then
      local fa fd
      fa="$(echo "$_ls_line" | sed 's/.*|+\([0-9]*\)|-.*/\1/')"
      fd="$(echo "$_ls_line" | sed 's/.*|-\([0-9]*\)$/\1/')"
      fa=${fa:-0}; fd=${fd:-0}
      if [[ $fa -gt 0 && $fd -gt 0 ]]; then
        stat_str="$(printf ' | %b+%d%b %b-%d%b' "$CLR_OK" "$fa" "$CLR_RST" "$CLR_ERR" "$fd" "$CLR_RST")"
      elif [[ $fa -gt 0 ]]; then
        stat_str="$(printf ' | %b+%d%b' "$CLR_OK" "$fa" "$CLR_RST")"
      elif [[ $fd -gt 0 ]]; then
        stat_str="$(printf ' | %b-%d%b' "$CLR_ERR" "$fd" "$CLR_RST")"
      fi
    fi
    printf '%s%s\n' "$l" "$stat_str"
  done

  # Summary line (git-style with line-level stats)
  local summary="${current} file(s) changed"
  if [[ "$total_line_adds" -gt 0 ]]; then
    summary+="$(printf ', %b%d insertion(+)%b' "$CLR_OK" "$total_line_adds" "$CLR_RST")"
  fi
  if [[ "$total_line_dels" -gt 0 ]]; then
    summary+="$(printf ', %b%d deletion(-)%b' "$CLR_ERR" "$total_line_dels" "$CLR_RST")"
  fi
  # Fallback: if no line stats, show file-level counts
  if [[ "$total_line_adds" -eq 0 && "$total_line_dels" -eq 0 ]]; then
    [[ "$new_count" -gt 0 ]] && summary+=", ${new_count} new"
    [[ "$del_count" -gt 0 ]] && summary+=", ${del_count} deleted"
    [[ "$mod_count" -gt 0 ]] && summary+=", ${mod_count} modified"
  fi
  echo "$summary"

  # Final stats: elapsed time + speed
  local total_speed_str=""
  if [[ $elapsed_s -gt 0 && $current -gt 0 ]]; then
    total_speed_str=" | $(printf '%.1f' "$(echo "scale=1; $current / $elapsed_s" | bc 2>/dev/null || echo "0")") files/s"
  fi
  printf 'Transfer complete. [%02d:%02d]%s ✔\n' "$mins" "$secs" "$total_speed_str"

  # Cleanup temp file
  rm -f "$_linestats_file" 2>/dev/null
}

# Multi-server wrapper with color-coded separators
# Args: direction(push|pull|fetch) target_servers_string
function multi_server_run() {
  local direction="$1"
  local target="$2"
  local delete_mode="$3"
  local servers=()
  local total=0
  local success=0
  local failed=0

  while IFS= read -r s; do
    [[ -z "$s" ]] && continue
    servers+=("$s")
    ((total++))
  done < <(each_target_server "$target")

  local idx=0
  for server in "${servers[@]}"; do
    ((idx++))
    local color emoji host
    color="$(server_color "$server")"
    emoji="$(server_emoji "$server")"
    host="$(resolve_host "$server")"

    local verb
    case "$direction" in
      push) verb="Pushing to" ;;
      pull) verb="Pulling from" ;;
      fetch) verb="Fetching from" ;;
    esac

    # Print separator
    printf '%b══════════════════════════════════════════════════%b\n' "$color" "$CLR_RST"
    printf '%b [%d/%d] %s %s .%s (%s)%b\n' "$color" "$idx" "$total" "$emoji" "$verb" "$server" "$host" "$CLR_RST"
    printf '%b══════════════════════════════════════════════════%b\n' "$color" "$CLR_RST"

    if git_style_transfer "$direction" "$server" "$delete_mode"; then
      ((success++))
    else
      ((failed++))
      printf '%b[ERROR] Transfer to .%s failed 🔴%b\n' "$CLR_ERR" "$server" "$CLR_RST"
    fi
    echo ""
  done

  if [[ "$total" -gt 1 ]]; then
    if [[ "$failed" -eq 0 ]]; then
      printf '══════════════════════════════════════════════════\n'
      printf '%b Summary: %d/%d servers completed successfully ✔%b\n' "$CLR_OK" "$success" "$total" "$CLR_RST"
      printf '══════════════════════════════════════════════════\n'
    else
      printf '══════════════════════════════════════════════════\n'
      printf '%b Summary: %d/%d completed, %d failed ✘%b\n' "$CLR_ERR" "$success" "$total" "$failed" "$CLR_RST"
      printf '══════════════════════════════════════════════════\n'
    fi
  fi
}

function die() {
  echo "[ERROR] $*" >&2
  exit 1
}

function note() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

function require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"
}

function ensure_password_tooling() {
  if [[ -n "$CFDLAB_PASSWORD" ]]; then
    command -v sshpass >/dev/null 2>&1 || die "CFDLAB_PASSWORD is set but sshpass is missing"
  fi
}

# ========== VPN Route Auto-Fix ==========
# macOS VPN often does not route 140.114.58.0/24 through the VPN tunnel.
# This function detects the issue and fixes it automatically.
CFDLAB_SUBNET="140.114.58.0/24"
CFDLAB_VPN_AUTOFIX="${CFDLAB_VPN_AUTOFIX:-1}"  # set to 0 to disable

function vpn_route_ok() {
  # Check if 140.114.58.87 routes through a ppp interface (VPN)
  local iface
  iface="$(route -n get 140.114.58.87 2>/dev/null | awk '/interface:/{print $2}')"
  [[ "$iface" == ppp* ]]
}

function vpn_is_connected() {
  # Check if any ppp interface is up (VPN connected)
  ifconfig 2>/dev/null | grep -q '^ppp[0-9]'
}

function vpn_fix_route() {
  # Find the first active ppp interface
  local ppp_iface
  ppp_iface="$(ifconfig 2>/dev/null | grep -o '^ppp[0-9]*' | head -1)"
  if [[ -z "$ppp_iface" ]]; then
    echo "[VPN] No VPN connection detected (no ppp interface)."
    return 1
  fi

  if vpn_route_ok; then
    echo "[VPN] Route OK — 140.114.58.0/24 → $ppp_iface"
    return 0
  fi

  echo "[VPN] Route missing — adding 140.114.58.0/24 → $ppp_iface ..."
  if sudo route add -net "$CFDLAB_SUBNET" -interface "$ppp_iface" >/dev/null 2>&1; then
    echo "[VPN] Route added successfully."
    return 0
  else
    echo "[VPN] Failed to add route (need sudo password)."
    return 1
  fi
}

function ensure_vpn_route() {
  # Silent auto-fix: only runs on macOS, only if VPN is up, only if route is wrong
  [[ "$(uname)" == "Darwin" ]] || return 0
  [[ "$CFDLAB_VPN_AUTOFIX" == "1" ]] || return 0
  vpn_is_connected || return 0  # no VPN, skip
  vpn_route_ok && return 0      # route already correct

  # Try to fix silently; if sudo needs password, prompt once
  local ppp_iface
  ppp_iface="$(ifconfig 2>/dev/null | grep -o '^ppp[0-9]*' | head -1)"
  [[ -n "$ppp_iface" ]] || return 0

  echo "[VPN] Auto-fixing route: 140.114.58.0/24 → $ppp_iface"
  sudo route add -net "$CFDLAB_SUBNET" -interface "$ppp_iface" >/dev/null 2>&1 || true
}
# ========== End VPN Route Auto-Fix ==========

function rsync_rsh_cmd() {
  if [[ -n "$CFDLAB_PASSWORD" ]]; then
    printf "sshpass -p '%s' ssh %s" "$CFDLAB_PASSWORD" "$CFDLAB_SSH_OPTS"
  else
    printf "ssh %s" "$CFDLAB_SSH_OPTS"
  fi
}

function ssh_batch_exec() {
  local host="$1"
  local remote_cmd="$2"

  if [[ -n "$CFDLAB_PASSWORD" ]]; then
    sshpass -p "$CFDLAB_PASSWORD" ssh ${CFDLAB_SSH_OPTS} "${CFDLAB_USER}@${host}" "$remote_cmd"
  else
    ssh ${CFDLAB_SSH_OPTS} "${CFDLAB_USER}@${host}" "$remote_cmd"
  fi
}

function normalize_server() {
  local raw="${1:-87}"
  raw="${raw#.}"
  case "$raw" in
    87|89|154) echo "$raw" ;;
    *) die "Unknown server '$1' (use 87, 89 or 154)" ;;
  esac
}

function resolve_host() {
  local server="$1"
  case "$server" in
    87) echo "140.114.58.87" ;;
    89) echo "140.114.58.89" ;;
    154) echo "140.114.58.154" ;;
    *) die "Unknown server '$server'" ;;
  esac
}

function parse_combo() {
  local combo="${1:-87:${CFDLAB_DEFAULT_NODE}}"
  local server
  local node

  if [[ "$combo" == *:* ]]; then
    server="${combo%%:*}"
    node="${combo##*:}"
  else
    server="$combo"
    node="${2:-$CFDLAB_DEFAULT_NODE}"
  fi

  server="$(normalize_server "$server")"
  [[ "$node" =~ ^[0-9]+$ ]] || die "Invalid node '$node'"
  echo "$server:$node"
}

function parse_server_or_all() {
  local raw="${1:-all}"
  raw="${raw#.}"
  if [[ "$raw" == "all" ]]; then
    echo "all"
    return
  fi
  echo "$(normalize_server "$raw")"
}

function each_target_server() {
  local target="$1"
  if [[ "$target" == "all" ]]; then
    echo "87"
    echo "89"
    echo "154"
  else
    echo "$target"
  fi
}

function run_on_node() {
  local server="$1"
  local node="$2"
  local remote_cmd="$3"
  local host

  host="$(resolve_host "$server")"
  # node=0 表示直連伺服器，不需要跳板到 cfdlab-ibX
  if [[ "$node" == "0" ]]; then
    ssh -t "${CFDLAB_USER}@${host}" "bash -lc 'cd ${CFDLAB_REMOTE_PATH}; $remote_cmd'"
  else
    ssh -t "${CFDLAB_USER}@${host}" "ssh -t cfdlab-ib${node} \"bash -lc '$remote_cmd'\""
  fi
}

function run_on_server() {
  local server="$1"
  local remote_cmd="$2"
  local host

  host="$(resolve_host "$server")"
  ssh -t "${CFDLAB_USER}@${host}" "bash -lc '$remote_cmd'"
}

# 確保遠端資料夾存在（自動建立）
function ensure_remote_dir() {
  local server="$1"
  local host

  host="$(resolve_host "$server")"
  note "Ensuring remote directory exists on ${server}: ${CFDLAB_REMOTE_PATH}"
  ssh_batch_exec "$host" "mkdir -p '${CFDLAB_REMOTE_PATH}'" 2>/dev/null || true
}

function push_args() {
  local delete_mode="$1"
  local rsh_cmd
  rsh_cmd="$(rsync_rsh_cmd)"
  local args=(
    -az
    # --- Protected directories: never synced, never deleted on remote ---
    --filter='P .git/'
    --exclude=.git/
    --filter='P .vscode/'
    --exclude=.vscode/
    --filter='P backup/'
    --exclude=backup/
    --filter='P result/'
    --exclude=result/
    --filter='P statistics/'
    --exclude=statistics/
    --filter='P __pycache__/'
    --exclude=__pycache__/
    # --- Build/data artifacts: not synced, deleted on remote with --delete-excluded ---
    --exclude=a.out
    --exclude=*.o
    --exclude=*.exe
    --exclude=*.dat
    --exclude=*.DAT
    --exclude=*.plt
    --exclude=*.bin
    --exclude=*.vtk
    --exclude=*.vtu
    --exclude=log*
    --exclude=*.swp
    --exclude=*.swo
    --exclude=*~
    --exclude=*.pyc
    --exclude=.DS_Store
    -e
    "$rsh_cmd"
  )

  if [[ "$delete_mode" == "delete" ]]; then
    args+=(--delete --delete-excluded)
  fi

  printf '%s\n' "${args[@]}"
}

function pull_args() {
  local delete_mode="$1"
  local rsh_cmd
  rsh_cmd="$(rsync_rsh_cmd)"
  local args=(
    -az
    --prune-empty-dirs
    --include=*/
    --include=*.dat
    --include=*.DAT
    --include=*.plt
    --include=*.bin
    --include=*.vtk
    --include=*.vtu
    --include=log*
    --exclude=*
    -e
    "$rsh_cmd"
  )

  if [[ "$delete_mode" == "delete" ]]; then
    args+=(--delete)
  fi

  printf '%s\n' "${args[@]}"
}

function build_arg_array() {
  local mode="$1"
  local delete_mode="$2"
  local out=()

  if [[ "$mode" == "push" ]]; then
    while IFS= read -r line; do
      out+=("$line")
    done < <(push_args "$delete_mode")
  else
    while IFS= read -r line; do
      out+=("$line")
    done < <(pull_args "$delete_mode")
  fi

  printf '%s\n' "${out[@]}"
}

function run_push() {
  local server="$1"
  local delete_mode="$2"
  local host
  local args=()

  host="$(resolve_host "$server")"
  
  # 自動建立遠端資料夾
  ensure_remote_dir "$server"
  
  while IFS= read -r line; do
    args+=("$line")
  done < <(build_arg_array push "$delete_mode")

  rsync "${args[@]}" "${WORKSPACE_DIR}/" "${CFDLAB_USER}@${host}:${CFDLAB_REMOTE_PATH}/"
}

function run_pull() {
  local server="$1"
  local delete_mode="$2"
  local host
  local args=()

  host="$(resolve_host "$server")"
  while IFS= read -r line; do
    args+=("$line")
  done < <(build_arg_array pull "$delete_mode")

  rsync "${args[@]}" "${CFDLAB_USER}@${host}:${CFDLAB_REMOTE_PATH}/" "${WORKSPACE_DIR}/"
}

# Helper: filter rsync output to keep only valid itemize lines
function filter_rsync_output() {
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    local flags="${line%% *}"
    # Keep only valid rsync itemize lines
    if [[ "$flags" =~ ^[.\<\>ch\*][fdLDS] ]] || [[ "$flags" == *deleting ]]; then
      printf '%s\n' "$line"
    fi
  done
}

function preview_push_changes() {
  local server="$1"
  local host
  local args=()
  local output

  host="$(resolve_host "$server")"
  while IFS= read -r line; do
    args+=("$line")
  done < <(build_arg_array push keep)
  args+=(--dry-run --itemize-changes)

  if ! output="$(rsync "${args[@]}" "${WORKSPACE_DIR}/" "${CFDLAB_USER}@${host}:${CFDLAB_REMOTE_PATH}/" 2>&1)"; then
    echo "$output" >&2
    return 1
  fi
  printf '%s\n' "$output" | filter_rsync_output
}

function preview_pull_changes() {
  local server="$1"
  local delete_mode="$2"
  local host
  local args=()
  local output

  host="$(resolve_host "$server")"
  while IFS= read -r line; do
    args+=("$line")
  done < <(build_arg_array pull "$delete_mode")
  args+=(--dry-run --itemize-changes)

  if ! output="$(rsync "${args[@]}" "${CFDLAB_USER}@${host}:${CFDLAB_REMOTE_PATH}/" "${WORKSPACE_DIR}/" 2>&1)"; then
    echo "$output" >&2
    return 1
  fi
  printf '%s\n' "$output" | filter_rsync_output
}

function count_change_lines() {
  local payload="$1"
  printf '%s\n' "$payload" | awk 'NF > 0 && $1 ~ /^[<>ch\.\*]/ && $1 !~ /^\.d/ {c++} END{print c+0}'
}

function list_change_paths() {
  local payload="$1"
  printf '%s\n' "$payload" | awk 'NF > 1 && $1 ~ /^[<>ch\.\*]/ {$1=""; sub(/^ /,""); print}'
}

function print_pending_summary() {
  local server="$1"
  local push_preview
  local pull_preview
  local fetch_preview

  if ! push_preview="$(preview_push_changes "$server")"; then
    printf '%s\n' "${server}: ERROR (push preview failed)"
    return
  fi
  if ! pull_preview="$(preview_pull_changes "$server" keep)"; then
    printf '%s\n' "${server}: ERROR (pull preview failed)"
    return
  fi
  if ! fetch_preview="$(preview_pull_changes "$server" delete)"; then
    printf '%s\n' "${server}: ERROR (fetch preview failed)"
    return
  fi

  printf '%s\n' "${server}: push=$(count_change_lines "$push_preview"), pull=$(count_change_lines "$pull_preview"), fetch=$(count_change_lines "$fetch_preview")"
}

function confirm_or_die() {
  local message="$1"

  if [[ "$CFDLAB_ASSUME_YES" == "1" ]]; then
    return
  fi

  if [[ ! -t 0 ]]; then
    die "$message (set CFDLAB_ASSUME_YES=1 for non-interactive mode)"
  fi

  read -r -p "$message [y/N]: " ans
  case "$ans" in
    y|Y|yes|YES) ;;
    *) die "Cancelled" ;;
  esac
}

# ========== Sync History Logging ==========
SYNC_HISTORY_FILE="${HOME}/.sync-history.log"

function log_sync_history() {
  local action="$1"    # PUSH / PULL / FETCH
  local server="$2"
  local file_count="$3"
  local adds="${4:-0}"
  local dels="${5:-0}"
  local timestamp
  timestamp="$(date '+%Y-%m-%d %H:%M')"
  printf '[%s] %s .%s: %s file(s), +%s -%s lines\n' \
    "$timestamp" "$action" "$server" "$file_count" "$adds" "$dels" \
    >> "$SYNC_HISTORY_FILE" 2>/dev/null || true
}

# ========== Code Diff Analysis Functions (GitHub-style) ==========

# Diff configuration defaults (can be overridden by .sync-config)
DIFF_CONTEXT_LINES=3
DIFF_MAX_FILE_SIZE=10485760     # 10 MB
DIFF_NEW_FILE_PREVIEW_LINES=20
DIFF_SMALL_DATA_MAX_LINES=50
DIFF_MAX_FILES_FULL=100
DIFF_IGNORE_WHITESPACE=true
SAFETY_DELETE_WARN=10
SAFETY_LINES_WARN=1000

# Load .sync-config if it exists
SYNC_CONFIG_FILE="${WORKSPACE_DIR}/.sync-config"
if [[ -f "$SYNC_CONFIG_FILE" ]]; then
  # Source only known safe variables
  while IFS='=' read -r key value; do
    key="$(echo "$key" | tr -d '[:space:]')"
    value="$(echo "$value" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | sed 's/^"//;s/"$//')"
    case "$key" in
      DIFF_CONTEXT_LINES) DIFF_CONTEXT_LINES="$value" ;;
      DIFF_MAX_FILE_SIZE) DIFF_MAX_FILE_SIZE="$value" ;;
      DIFF_NEW_FILE_PREVIEW_LINES) DIFF_NEW_FILE_PREVIEW_LINES="$value" ;;
      DIFF_SMALL_DATA_MAX_LINES) DIFF_SMALL_DATA_MAX_LINES="$value" ;;
      DIFF_MAX_FILES_FULL) DIFF_MAX_FILES_FULL="$value" ;;
      DIFF_IGNORE_WHITESPACE) DIFF_IGNORE_WHITESPACE="$value" ;;
      SAFETY_DELETE_WARN) SAFETY_DELETE_WARN="$value" ;;
      SAFETY_LINES_WARN) SAFETY_LINES_WARN="$value" ;;
      SYNC_HISTORY_FILE) SYNC_HISTORY_FILE="$value" ;;
    esac
  done < <(grep -v '^#' "$SYNC_CONFIG_FILE" | grep -v '^$' | grep '=')
fi

# Known text extensions for full diff display
DIFF_TEXT_EXT_PATTERN='\.(cpp|c|h|hpp|cu|cuh|py|sh|zsh|ps1|f90|f95|f03|mk|txt|md|rst|json|yaml|yml|toml)$|^Makefile$'

function is_text_file() {
  local filename="$1"
  local base
  base="$(basename "$filename")"
  if echo "$base" | grep -qiE "$DIFF_TEXT_EXT_PATTERN"; then
    return 0
  fi
  return 1
}

function is_binary_file() {
  local filename="$1"
  local base
  base="$(basename "$filename")"
  if echo "$base" | grep -qiE '\.(o|out|exe|bin|vtk|vtu|dat|DAT|plt|png|jpg|gif|pdf|zip|tar|gz|bz2)$'; then
    return 0
  fi
  return 1
}

# Get remote file size in bytes
function get_remote_file_size() {
  local server="$1"
  local filepath="$2"
  local host
  host="$(resolve_host "$server")"
  ssh_batch_exec "$host" "stat -c%s '${CFDLAB_REMOTE_PATH}/${filepath}' 2>/dev/null || echo 0" 2>/dev/null
}

# Get local file size in bytes
function get_local_file_size() {
  local filepath="$1"
  local full_path="${WORKSPACE_DIR}/${filepath}"
  if [[ -f "$full_path" ]]; then
    stat -f%z "$full_path" 2>/dev/null || stat -c%s "$full_path" 2>/dev/null || echo 0
  else
    echo 0
  fi
}

# Format file size for display
function format_size() {
  local bytes="$1"
  if [[ "$bytes" -ge 1073741824 ]]; then
    printf '%.1f GB' "$(echo "scale=1; $bytes / 1073741824" | bc)"
  elif [[ "$bytes" -ge 1048576 ]]; then
    printf '%.1f MB' "$(echo "scale=1; $bytes / 1048576" | bc)"
  elif [[ "$bytes" -ge 1024 ]]; then
    printf '%.1f KB' "$(echo "scale=1; $bytes / 1024" | bc)"
  else
    printf '%d B' "$bytes"
  fi
}

# Fetch remote file content to a temp file (returns temp path)
function fetch_remote_to_temp() {
  local server="$1"
  local filepath="$2"
  local host
  local tmpfile
  host="$(resolve_host "$server")"
  tmpfile="$(mktemp)"
  ssh_batch_exec "$host" "cat '${CFDLAB_REMOTE_PATH}/${filepath}'" > "$tmpfile" 2>/dev/null
  echo "$tmpfile"
}

# Generate a unified diff between local and remote file and print GitHub-style output.
# Returns total +lines and -lines via global variables.
DIFF_TOTAL_ADDS=0
DIFF_TOTAL_DELS=0

function show_file_diff() {
  local direction="$1"   # push or pull
  local server="$2"
  local filepath="$3"
  local file_status="$4" # new / modified / deleted
  local mode="${5:-full}" # full / summary / stat

  local local_file="${WORKSPACE_DIR}/${filepath}"
  local adds=0 dels=0
  local base
  base="$(basename "$filepath")"

  # ── New file (exists only on source side) ──
  if [[ "$file_status" == "new" ]]; then
    # For pull/fetch: new file is on remote, not local yet
    local is_remote_new=0
    [[ "$direction" == "pull" || "$direction" == "fetch" ]] && is_remote_new=1

    if [[ "$mode" == "stat" ]]; then
      if [[ "$is_remote_new" -eq 1 ]]; then
        local host_tmp
        host_tmp="$(resolve_host "$server")"
        adds=$(ssh_batch_exec "$host_tmp" "wc -l < '${CFDLAB_REMOTE_PATH}/${filepath}'" 2>/dev/null | tr -d ' ')
      else
        adds=$(wc -l < "$local_file" 2>/dev/null || echo 0)
      fi
      adds="${adds##*( )}"
      adds="${adds:-0}"
      printf '📄 %-50s \033[32m+%s\033[0m (new file)\n' "$filepath" "$adds"
      DIFF_TOTAL_ADDS=$((DIFF_TOTAL_ADDS + adds))
      return
    fi

    if is_text_file "$filepath"; then
      if [[ "$is_remote_new" -eq 1 ]]; then
        local host_tmp
        host_tmp="$(resolve_host "$server")"
        adds=$(ssh_batch_exec "$host_tmp" "wc -l < '${CFDLAB_REMOTE_PATH}/${filepath}'" 2>/dev/null | tr -d ' ')
        adds="${adds##*( )}"
        adds="${adds:-0}"
        printf '\n\033[34m📄 %s\033[0m %50s \033[32m+%s -0\033[0m (new file)\n' \
          "$filepath" "" "$adds"
        printf '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n'
        printf '\033[33m@@ -0,0 +1,%s @@\033[0m\n' "$adds"
        local remote_tmp
        remote_tmp="$(fetch_remote_to_temp "$server" "$filepath")"
        if [[ -f "$remote_tmp" ]]; then
          head -n "$DIFF_NEW_FILE_PREVIEW_LINES" "$remote_tmp" | while IFS= read -r line; do
            printf '\033[32m+%s\033[0m\n' "$line"
          done
          local total_lines
          total_lines=$(wc -l < "$remote_tmp" 2>/dev/null || echo 0)
          total_lines="${total_lines##*( )}"
          if [[ "$total_lines" -gt "$DIFF_NEW_FILE_PREVIEW_LINES" ]]; then
            local remaining=$((total_lines - DIFF_NEW_FILE_PREVIEW_LINES))
            printf '\033[90m... (%d more lines)\033[0m\n' "$remaining"
          fi
          rm -f "$remote_tmp"
        fi
      else
        adds=$(wc -l < "$local_file" 2>/dev/null || echo 0)
        adds="${adds##*( )}"
        printf '\n\033[34m📄 %s\033[0m %50s \033[32m+%s -0\033[0m (new file)\n' \
          "$filepath" "" "$adds"
        printf '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n'
        printf '\033[33m@@ -0,0 +1,%s @@\033[0m\n' "$adds"
        head -n "$DIFF_NEW_FILE_PREVIEW_LINES" "$local_file" | while IFS= read -r line; do
          printf '\033[32m+%s\033[0m\n' "$line"
        done
        local total_lines
        total_lines=$(wc -l < "$local_file" 2>/dev/null || echo 0)
        total_lines="${total_lines##*( )}"
        if [[ "$total_lines" -gt "$DIFF_NEW_FILE_PREVIEW_LINES" ]]; then
          local remaining=$((total_lines - DIFF_NEW_FILE_PREVIEW_LINES))
          printf '\033[90m... (%d more lines)\033[0m\n' "$remaining"
        fi
      fi
    else
      local fsize
      if [[ "$is_remote_new" -eq 1 ]]; then
        fsize=$(get_remote_file_size "$server" "$filepath")
      else
        fsize=$(get_local_file_size "$filepath")
      fi
      adds=1
      printf '\n\033[34m📄 %s\033[0m (new file, %s)\n' "$filepath" "$(format_size "$fsize")"
    fi
    DIFF_TOTAL_ADDS=$((DIFF_TOTAL_ADDS + adds))
    return
  fi

  # ── Deleted file ──
  if [[ "$file_status" == "deleted" ]]; then
    if [[ "$mode" == "stat" ]]; then
      printf '📄 %-50s \033[31m(deleted)\033[0m\n' "$filepath"
      return
    fi
    printf '\n\033[34m📄 %s\033[0m \033[31m(will be deleted)\033[0m\n' "$filepath"
    return
  fi

  # ── Modified file: need actual diff ──
  if is_binary_file "$filepath"; then
    local local_size remote_size
    local_size=$(get_local_file_size "$filepath")
    remote_size=$(get_remote_file_size "$server" "$filepath")
    printf '\n\033[34m📄 %s\033[0m (binary: %s → %s)\n' \
      "$filepath" "$(format_size "$remote_size")" "$(format_size "$local_size")"
    return
  fi

  if ! is_text_file "$filepath"; then
    # Non-text, non-binary: check size
    local fsize
    fsize=$(get_local_file_size "$filepath")
    if [[ "$fsize" -gt "$DIFF_MAX_FILE_SIZE" ]]; then
      local remote_size
      remote_size=$(get_remote_file_size "$server" "$filepath")
      printf '\n\033[34m📄 %s\033[0m (large: %s → %s)\n' \
        "$filepath" "$(format_size "$remote_size")" "$(format_size "$fsize")"
      return
    fi
  fi

  # Fetch remote file for comparison
  local remote_tmp
  remote_tmp="$(fetch_remote_to_temp "$server" "$filepath")"

  local diff_args=(-u --label "a/$filepath" --label "b/$filepath")
  if [[ "$DIFF_IGNORE_WHITESPACE" == "true" ]]; then
    diff_args+=(-w)
  fi

  local diff_output
  local old_file new_file
  if [[ "$direction" == "push" ]]; then
    old_file="$remote_tmp"
    new_file="$local_file"
  else
    old_file="$local_file"
    new_file="$remote_tmp"
  fi

  diff_output=$(diff "${diff_args[@]}" "$old_file" "$new_file" 2>/dev/null || true)

  # Count adds/dels
  adds=$(echo "$diff_output" | grep -c '^+[^+]' 2>/dev/null) || adds=0
  dels=$(echo "$diff_output" | grep -c '^-[^-]' 2>/dev/null) || dels=0

  DIFF_TOTAL_ADDS=$((DIFF_TOTAL_ADDS + adds))
  DIFF_TOTAL_DELS=$((DIFF_TOTAL_DELS + dels))

  if [[ "$mode" == "stat" ]]; then
    printf '📄 %-50s \033[32m+%s\033[0m \033[31m-%s\033[0m\n' "$filepath" "$adds" "$dels"
    rm -f "$remote_tmp"
    return
  fi

  # GitHub-style header
  printf '\n\033[34m📄 %s\033[0m %50s \033[32m+%s\033[0m \033[31m-%s\033[0m\n' \
    "$filepath" "" "$adds" "$dels"
  printf '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n'

  if [[ -z "$diff_output" ]]; then
    printf '\033[90m  (no content changes — whitespace only?)\033[0m\n'
    rm -f "$remote_tmp"
    return
  fi

  # Print diff with colors (skip first 2 header lines from unified diff as we print our own)
  local line_no=0
  local in_small_data=0
  local small_data_lines=0
  echo "$diff_output" | while IFS= read -r line; do
    ((line_no++))
    # Skip unified diff header lines (--- a/... and +++ b/...)
    [[ "$line_no" -le 2 ]] && continue

    # For non-text files, limit output
    if ! is_text_file "$filepath"; then
      ((small_data_lines++))
      if [[ "$small_data_lines" -gt "$DIFF_SMALL_DATA_MAX_LINES" ]]; then
        printf '\033[90m... (truncated, showing first %d diff lines)\033[0m\n' "$DIFF_SMALL_DATA_MAX_LINES"
        break
      fi
    fi

    case "$line" in
      @@*)
        printf '\033[33m%s\033[0m\n' "$line"
        ;;
      +*)
        printf '\033[32m%s\033[0m\n' "$line"
        ;;
      -*)
        printf '\033[31m%s\033[0m\n' "$line"
        ;;
      *)
        printf '\033[90m%s\033[0m\n' "$line"
        ;;
    esac
  done

  rm -f "$remote_tmp"
}

# Full code diff analysis for a server before push/pull/fetch
# Args: direction server [mode] 
#   mode: full (default), summary, stat, no-diff
# Outputs: colored diff to stdout
# Sets: DIFF_TOTAL_ADDS, DIFF_TOTAL_DELS, DIFF_FILE_COUNT
# Returns: 0 if user confirms (or no confirm needed), 1 if cancelled
DIFF_FILE_COUNT=0

function analyze_code_diff() {
  local direction="$1"
  local server="$2"
  local mode="${3:-full}"
  local confirm="${4:-false}"

  DIFF_TOTAL_ADDS=0
  DIFF_TOTAL_DELS=0
  DIFF_FILE_COUNT=0

  if [[ "$mode" == "no-diff" ]]; then
    return 0
  fi

  local host
  host="$(resolve_host "$server")"

  printf '\n\033[1m🔍 Analyzing changes before %s...\033[0m\n' "$direction"
  printf '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n'

  # Get changed files from rsync dry-run
  local preview_raw=""
  if [[ "$direction" == "push" ]]; then
    preview_raw="$(preview_push_changes "$server" 2>/dev/null)" || {
      printf '\033[31m[ERROR] Failed to enumerate changes\033[0m\n'
      return 0
    }
  else
    local dm="keep"
    [[ "$direction" == "fetch" ]] && dm="delete"
    preview_raw="$(preview_pull_changes "$server" "$dm" 2>/dev/null)" || {
      printf '\033[31m[ERROR] Failed to enumerate changes\033[0m\n'
      return 0
    }
  fi

  # Parse rsync itemize output into arrays
  local new_files=() modified_files=() deleted_files=()
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    local flags="${line%% *}"
    local path="${line#* }"
    path="${path#./}"
    [[ -z "$path" || "$path" == "." || "$path" == "./" ]] && continue

    local cls
    cls="$(classify_rsync_flags "$flags")"
    case "$cls" in
      new) new_files+=("$path") ;;
      deleted) deleted_files+=("$path") ;;
      modified) modified_files+=("$path") ;;
      skip) ;; # directories, ignore
    esac
  done <<< "$preview_raw"

  local total_new=${#new_files[@]}
  local total_mod=${#modified_files[@]}
  local total_del=${#deleted_files[@]}
  local total_files=$((total_new + total_mod + total_del))
  DIFF_FILE_COUNT=$total_files

  if [[ "$total_files" -eq 0 ]]; then
    printf '\n  Already up to date.\n\n'
    return 0
  fi

  # ── Changes Summary ──
  printf '\n\033[1m📊 Changes Summary:\033[0m\n'
  [[ "$total_mod" -gt 0 ]] && printf '   📝 Modified: %d files\n' "$total_mod"
  [[ "$total_new" -gt 0 ]] && printf '   ✨ New: %d files\n' "$total_new"
  [[ "$total_del" -gt 0 ]] && printf '   🗑️  Deleted: %d files\n' "$total_del"
  printf '\n'

  # ── Safety warnings ──
  if [[ "$total_del" -ge "$SAFETY_DELETE_WARN" ]]; then
    printf '\033[31m⚠️  WARNING: %d files will be deleted!\033[0m\n' "$total_del"
  fi

  # If too many files, show only summary
  if [[ "$total_files" -gt "$DIFF_MAX_FILES_FULL" ]]; then
    printf '\033[90m(Too many files (%d) for detailed diff — showing stat only)\033[0m\n\n' "$total_files"
    mode="stat"
  fi

  # ── Per-file diff ──
  if [[ "$mode" != "summary" ]]; then
    # Modified files
    for filepath in ${modified_files[@]+"${modified_files[@]}"}; do
      show_file_diff "$direction" "$server" "$filepath" "modified" "$mode"
    done

    # New files
    for filepath in ${new_files[@]+"${new_files[@]}"}; do
      show_file_diff "$direction" "$server" "$filepath" "new" "$mode"
    done

    # Deleted files
    for filepath in ${deleted_files[@]+"${deleted_files[@]}"}; do
      show_file_diff "$direction" "$server" "$filepath" "deleted" "$mode"
    done
  fi

  # ── Code Changes Statistics ──
  printf '\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n'
  printf '\033[1m📊 Code Changes Statistics:\033[0m\n'
  [[ "$DIFF_TOTAL_ADDS" -gt 0 ]] && printf '   📈 Lines added: %d (+)\n' "$DIFF_TOTAL_ADDS"
  [[ "$DIFF_TOTAL_DELS" -gt 0 ]] && printf '   📉 Lines removed: %d (-)\n' "$DIFF_TOTAL_DELS"
  local net=$((DIFF_TOTAL_ADDS - DIFF_TOTAL_DELS))
  printf '   📐 Net change: %+d lines\n' "$net"

  if [[ "$((DIFF_TOTAL_ADDS + DIFF_TOTAL_DELS))" -ge "$SAFETY_LINES_WARN" ]]; then
    printf '\n\033[33m⚠️  Large change: %d total line changes\033[0m\n' "$((DIFF_TOTAL_ADDS + DIFF_TOTAL_DELS))"
  fi

  # ── File type breakdown ──
  # Group by extension (using sort/uniq for portability)
  local ext_list=""
  for filepath in ${modified_files[@]+"${modified_files[@]}"} ${new_files[@]+"${new_files[@]}"}; do
    local ext="${filepath##*.}"
    if [[ "$filepath" == *.* ]]; then
      ext_list="${ext_list}.${ext}"$'\n'
    else
      ext_list="${ext_list}(no-ext)"$'\n'
    fi
  done
  if [[ -n "$ext_list" ]]; then
    local ext_summary
    ext_summary=$(printf '%s' "$ext_list" | sort | uniq -c | sort -rn)
    if [[ -n "$ext_summary" ]]; then
      printf '\n   📁 Changed files by type:\n'
      while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        local count ext_name
        count=$(echo "$line" | awk '{print $1}')
        ext_name=$(echo "$line" | awk '{print $2}')
        printf '      %s: %d file(s)\n' "$ext_name" "$count"
      done <<< "$ext_summary"
    fi
  fi

  printf '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n'

  # ── Confirmation ──
  if [[ "$confirm" == "true" ]]; then
    printf '\033[33m⚠️  Review changes above before %s\033[0m\n' "$direction"
    if [[ ! -t 0 ]] || [[ "$CFDLAB_ASSUME_YES" == "1" ]]; then
      return 0
    fi
    read -r -p "Continue with $direction? [Y/n]: " ans
    case "$ans" in
      n|N|no|NO) printf 'Cancelled.\n'; return 1 ;;
      *) return 0 ;;
    esac
  fi

  return 0
}

# ========== End Code Diff Analysis ==========

function watch_pid_file() {
  case "$1" in
    push) echo "$WATCHPUSH_PID" ;;
    pull) echo "$WATCHPULL_PID" ;;
    fetch) echo "$WATCHFETCH_PID" ;;
    vtkrename) echo "$VTKRENAME_PID" ;;
    *) die "Unknown watch kind: $1" ;;
  esac
}

function watch_log_file() {
  case "$1" in
    push) echo "$WATCHPUSH_LOG" ;;
    pull) echo "$WATCHPULL_LOG" ;;
    fetch) echo "$WATCHFETCH_LOG" ;;
    vtkrename) echo "$VTKRENAME_LOG" ;;
    *) die "Unknown watch kind: $1" ;;
  esac
}

function is_pid_alive() {
  local pid="$1"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" >/dev/null 2>&1
}

function read_pid_file() {
  local pid_file="$1"
  [[ -f "$pid_file" ]] || return 0
  tr -cd '0-9' <"$pid_file"
}

function daemon_loop() {
  local kind="$1"
  local target="$2"
  local interval="$3"

  note "${kind} daemon started target=${target} interval=${interval}s"
  while true; do
    case "$kind" in
      push) cmd_autopush "$target" ;;
      pull) cmd_autopull "$target" ;;
      fetch) cmd_autofetch "$target" ;;
      vtkrename) run_vtkrename_once ;;
      *) die "Unknown daemon kind: $kind" ;;
    esac
    sleep "$interval"
  done
}

function start_daemon() {
  local kind="$1"
  local target="$2"
  local interval="$3"
  local pid_file
  local log_file

  [[ "$interval" =~ ^[0-9]+$ ]] || die "Interval must be an integer"
  mkdir -p "$STATE_DIR"

  pid_file="$(watch_pid_file "$kind")"
  log_file="$(watch_log_file "$kind")"

  if [[ -f "$pid_file" ]]; then
    local old_pid
    old_pid="$(read_pid_file "$pid_file")"
    if is_pid_alive "$old_pid"; then
      echo "[RUNNING] ${kind} daemon PID=${old_pid}"
      return
    fi
  fi

  nohup bash "$0" __daemon_loop "$kind" "$target" "$interval" >>"$log_file" 2>&1 &
  echo $! >"$pid_file"
  echo "[STARTED] ${kind} daemon PID=$!"
}

function stop_daemon() {
  local kind="$1"
  local pid_file
  pid_file="$(watch_pid_file "$kind")"

  if [[ ! -f "$pid_file" ]]; then
    echo "[STOPPED] ${kind} daemon not running"
    return
  fi

  local pid
  pid="$(read_pid_file "$pid_file")"
  if is_pid_alive "$pid"; then
    kill "$pid" >/dev/null 2>&1 || true
    sleep 0.2
    if is_pid_alive "$pid"; then
      kill -9 "$pid" >/dev/null 2>&1 || true
    fi
    echo "[STOPPED] ${kind} daemon PID=${pid}"
  else
    echo "[STOPPED] ${kind} daemon already inactive"
  fi

  rm -f "$pid_file"
}

function status_daemon() {
  local kind="$1"
  local pid_file
  local log_file

  pid_file="$(watch_pid_file "$kind")"
  log_file="$(watch_log_file "$kind")"

  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(read_pid_file "$pid_file")"
    if is_pid_alive "$pid"; then
      echo "[RUNNING] ${kind} daemon PID=${pid}"
      return
    fi
  fi

  echo "[STOPPED] ${kind} daemon"
}

function show_daemon_log() {
  local kind="$1"
  local log_file
  log_file="$(watch_log_file "$kind")"

  if [[ ! -f "$log_file" ]]; then
    echo "No log for ${kind}"
    return
  fi

  tail -n 50 "$log_file"
}

function clear_daemon_log() {
  local kind="$1"
  local log_file
  log_file="$(watch_log_file "$kind")"
  : >"$log_file"
  echo "[CLEARED] ${kind} log"
}

function run_vtkrename_once() {
  local result_dir="$WORKSPACE_DIR/result"
  [[ -d "$result_dir" ]] || return

  local file
  for file in "$result_dir"/velocity_merged_*.vtk; do
    [[ -e "$file" ]] || continue

    local base
    base="$(basename "$file")"
    if [[ "$base" =~ ^velocity_merged_([0-9]+)\.vtk$ ]]; then
      local step
      step="${BASH_REMATCH[1]}"
      if (( ${#step} < 6 )); then
        local padded
        local target
        padded="$(printf '%06d' "$step")"
        target="$result_dir/velocity_merged_${padded}.vtk"
        if [[ ! -e "$target" ]]; then
          mv "$file" "$target"
          note "VTK renamed: ${base} -> velocity_merged_${padded}.vtk"
        fi
      fi
    fi
  done
}

function cmd_help() {
  cat <<'EOF'
Mac commands (Windows-compatible names)

Core (未指定伺服器 = all):
  push [server]           - 上傳（預設 all）
  pull [server]           - 下載（預設 all）
  fetch [server]          - 下載+刪除本地多餘（預設 all）
  log [server]            - 查看遠端 log（預設 all）
  diff [server]           - 查看差異（預設 all）
  autopush [server]       - 自動上傳（預設 all）
  autopull [server]       - 自動下載（預設 all）
  autofetch [server]      - 自動 fetch（預設 all）
  watchpush [server]      - 背景持續上傳（預設 all）
  watchpull [server]      - 背景持續下載（預設 all）
  watchfetch [server]     - 背景持續 fetch（預設 all）
  check, status, add, reset, delete, clone
  sync, fullsync, issynced, syncstatus, bgstatus, vtkrename

Server shortcuts (指定單台):
  pull87, pull89, pull154, fetch87, fetch89, fetch154
  push87, push89, push154, pushall
  autopull87, autopull89, autopull154
  autofetch87, autofetch89, autofetch154
  autopush87, autopush89, autopush154, autopushall
  diff87, diff89, diff154, diffall
  log87, log89, log154

Code Diff Analysis (GitHub-style):
  sync-diff [server]     - 僅比較差異，不同步
  sync-diff-summary [s]  - 快速摘要（僅統計）
  sync-diff-file <file>  - 檢視特定檔案差異
  sync-log [lines]       - 查看同步歷史記錄
  sync-stop              - 停止所有背景同步任務

Diff options (for push/pull/fetch):
  --no-diff              - 跳過差異分析，直接傳輸
  --diff-summary         - 僅顯示統計摘要
  --diff-stat            - diffstat 風格統計
  --diff-full            - 完整差異（預設）
  --force                - 跳過確認和差異分析
  --quick                - 同 --no-diff

Extra node helpers:
  ssh [87:3], run [87:3] [gpu], jobs [87:3], kill [87:3]

GPU Status:
  gpus              - GPU 狀態總覽（所有伺服器）
  gpu [89|87|154]   - 詳細 GPU 狀態（nvidia-smi 完整輸出）

Watch subcommands:
  <watchcmd> status | log | stop | clear
  <watchcmd> [server|all] [interval]

Optional environment:
  CFDLAB_PASSWORD=<password>   (requires sshpass for password mode)
  CFDLAB_ASSUME_YES=1          (skip confirmations for reset/clone/sync/fullsync)

Examples:
  mobaxterm push                  # 上傳到所有伺服器（預設 all）
  mobaxterm push 87               # 僅上傳到 .87
  mobaxterm push --no-diff        # 直接上傳不顯示差異
  mobaxterm push --diff-stat 87   # 僅統計後上傳到 .87
  mobaxterm pull                  # 從所有伺服器下載（預設 all）
  mobaxterm pull 89               # 僅從 .89 下載
  mobaxterm pull --quick          # 快速下載不分析
  mobaxterm autopull              # 自動下載所有伺服器
  mobaxterm autopull 87           # 自動下載僅 .87
  mobaxterm watchfetch            # 背景監控所有伺服器
  mobaxterm watchfetch 154        # 背景監控僅 .154
  mobaxterm sync-diff 87          # 僅查看 .87 的差異
  mobaxterm sync-diff-file main.cu  # 查看特定檔案差異
  mobaxterm sync-log              # 查看同步歷史
  mobaxterm sync-stop             # 停止所有背景任務
EOF
}

# ── 互動式 SSH：上方選單 (即時) + 下方 GPU 狀態 (按 Enter 載入) ──
function cmd_issh() {
  local mode="${1:-switch}"   # switch | reconnect

  # 節點定義: "server:node:label:gpu_type"
  local -a NODES=(
    "89:0:.89  直連:V100-32G"
    "87:2:.87→ib2:P100-16G"
    "87:3:.87→ib3:P100-16G"
    "87:5:.87→ib5:P100-16G"
    "87:6:.87→ib6:V100-16G"
    "154:1:.154→ib1:P100-16G"
    "154:4:.154→ib4:P100-16G"
    "154:7:.154→ib7:P100-16G"
    "154:9:.154→ib9:P100-16G"
  )
  local total=${#NODES[@]}

  # 解析 nvidia-smi 輸出 → "dots|free|total"
  _parse_gpus() {
    local file="$1"
    if [[ ! -s "$file" ]] || grep -q "OFFLINE" "$file" 2>/dev/null; then
      echo "OFFLINE|0|0"; return
    fi
    local total_g=0 free_g=0 dots=""
    while IFS=',' read -r _idx mem_used mem_total util; do
      [[ -z "$_idx" ]] && continue
      util="${util//[^0-9]/}"; mem_used="${mem_used//[^0-9]/}"
      [[ -z "$util" ]] && continue
      ((total_g++))
      if (( util < 10 && mem_used < 100 )); then
        ((free_g++)); dots="${dots}🟢"
      else
        dots="${dots}🔴"
      fi
    done < "$file"
    [[ "$total_g" -eq 0 ]] && echo "OFFLINE|0|0" || echo "${dots}|${free_g}|${total_g}"
  }

  # ══════════════════════════════════════════════════
  # ① 立即顯示選單 (上方)
  # ══════════════════════════════════════════════════
  echo ""
  echo " ╔═══════════════════════════════════════╗"
  echo " ║       🖥  SSH 節點選擇                ║"
  echo " ╠═══╤═══════════╤═════════════════════╣"
  local idx=0
  for entry in "${NODES[@]}"; do
    local rest="${entry#*:}"; rest="${rest#*:}"
    local label="${rest%%:*}"; local gtype="${rest##*:}"
    ((idx++))
    printf " ║ %d │ %-9s │ %-8s            ║\n" "$idx" "$label" "$gtype"
  done
  echo " ╠═══╧═══════════╧═════════════════════╣"
  echo " ║ 0 │ 取消                             ║"
  echo " ╚═══╧═════════════════════════════════╝"
  echo ""
  echo "  💡 輸入編號 → 立即連線"
  echo "     按 Enter → 先查看 GPU 使用狀態再選"
  echo ""

  # ══════════════════════════════════════════════════
  # ② 背景啟動 GPU 查詢 (不阻塞選單)
  # ══════════════════════════════════════════════════
  local tmpdir
  tmpdir="$(mktemp -d)"

  ( sshpass -p "$CFDLAB_PASSWORD" \
      ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new \
      "${CFDLAB_USER}@140.114.58.89" \
      "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader" \
      2>/dev/null > "$tmpdir/89_0" || echo "OFFLINE" > "$tmpdir/89_0"
  ) &
  for n in 2 3 5 6; do
    ( ssh_batch_exec "140.114.58.87" \
        "ssh -o ConnectTimeout=5 cfdlab-ib${n} 'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader'" \
        2>/dev/null > "$tmpdir/87_${n}" || echo "OFFLINE" > "$tmpdir/87_${n}"
    ) &
  done
  for n in 1 4 7 9; do
    ( ssh_batch_exec "140.114.58.154" \
        "ssh -o ConnectTimeout=5 cfdlab-ib${n} 'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader'" \
        2>/dev/null > "$tmpdir/154_${n}" || echo "OFFLINE" > "$tmpdir/154_${n}"
    ) &
  done

  # ══════════════════════════════════════════════════
  # ③ 等待使用者輸入
  # ══════════════════════════════════════════════════
  local choice
  read -rp "  選擇 [1-${total}]: " choice

  # 使用者按 Enter (空白) → 等待 GPU 查完 → 顯示狀態 → 再選
  if [[ -z "$choice" ]]; then
    echo ""
    echo "  ⏳ 正在查詢 GPU 狀態..."
    wait

    echo ""
    echo " ── GPU 使用狀況 (僅供參考) ────────────────────"
    printf "  %-11s %-8s  %-24s %s\n" "Server" "GPU" "0 1 2 3 4 5 6 7" "Free"
    printf "  %-11s %-8s  %-24s %s\n" "───────────" "────────" "────────────────────" "────"

    for entry in "${NODES[@]}"; do
      local srv="${entry%%:*}"
      local rest="${entry#*:}"; local nd="${rest%%:*}"
      rest="${rest#*:}"; local label="${rest%%:*}"; local gtype="${rest##*:}"

      local result; result="$(_parse_gpus "$tmpdir/${srv}_${nd}")"
      local dots="${result%%|*}"
      local tmp="${result#*|}"; local nfree="${tmp%%|*}"; local ntotal="${tmp##*|}"

      if [[ "$dots" == "OFFLINE" ]]; then
        printf '  %-11s %-8s  ⬛⬛⬛⬛⬛⬛⬛⬛  OFFLINE\n' "$label" "$gtype"
      else
        printf '  %-11s %-8s  %s  %s/%s\n' "$label" "$gtype" "$dots" "$nfree" "$ntotal"
      fi
    done
    echo "  ─────────── ──────── ──────────────────── ────"
    echo ""
    echo " 🟢=閒置  🔴=使用中  ⬛=離線"
    echo ""

    read -rp "  請參考上方選單輸入編號 [1-${total}, 0=取消]: " choice
  fi

  # 背景任務清理
  rm -rf "$tmpdir"

  # ══════════════════════════════════════════════════
  # ④ 處理選擇
  # ══════════════════════════════════════════════════
  if [[ -z "$choice" ]] || [[ "$choice" == "0" ]]; then
    echo "已取消。"
    return 0
  fi

  if ! [[ "$choice" =~ ^[0-9]+$ ]] || (( choice < 1 || choice > total )); then
    die "無效的選擇: $choice"
  fi

  local selected="${NODES[$((choice-1))]}"
  local combo_srv="${selected%%:*}"
  local combo_rest="${selected#*:}"
  local combo_nd="${combo_rest%%:*}"
  local combo="${combo_srv}:${combo_nd}"

  echo ""
  note "連線到: ${combo}"
  cmd_ssh "$combo"
}

# ── VS Code QuickPick 模式：從 tasks.json 的 input:sshNodePicker 接收選擇 ──
function cmd_issh_quick() {
  local combo="${1:-}"
  if [[ -z "$combo" || "$combo" == "gpus" ]]; then
    # 使用者選了「先查 GPU 再選」→ 顯示 GPU 狀態 → 進入終端選單
    cmd_gpus
    echo ""
    echo "  📋 請參考上方 GPU 狀態，選擇節點連線："
    echo ""
    cmd_issh
  else
    # 使用者在 QuickPick 直接選了節點 → 直接連線
    cmd_ssh "$combo"
  fi
}

function cmd_ssh() {
  local parsed
  local server
  local node
  local host

  parsed="$(parse_combo "${1:-87:${CFDLAB_DEFAULT_NODE}}")"
  server="${parsed%%:*}"
  node="${parsed##*:}"
  host="$(resolve_host "$server")"

  # === 顯示目標節點的 GPU 狀態 ===
  echo ""
  echo "╔══════════════════════════════════════════════════════════════════╗"
  echo "║  🔗 準備連線到: .${server} $([ "$node" != "0" ] && echo "ib${node}" || echo "(直連)")                                    ║"
  echo "╠══════════════════════════════════════════════════════════════════╣"
  
  local gpu_output
  local free_status
  
  if [[ "$node" == "0" ]]; then
    # 直連伺服器 (如 .89)
    echo "║  📍 .${server} (${host})                                           ║"
    if [[ "$server" == "89" ]]; then
      echo "║  ├─ GPU: 8× Tesla V100-SXM2-32GB                                 ║"
    else
      echo "║  ├─ GPU: 8× Tesla P100-PCIE-16GB                                 ║"
    fi
    
    if gpu_output="$(ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new "${CFDLAB_USER}@${host}" "nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader" 2>/dev/null)"; then
      free_status=$(get_gpu_status_line "$gpu_output")
      local free_count="${free_status%%/*}"
      if [[ "$free_count" -eq 0 ]]; then
        echo "║  └─ 狀態: ❌ 全部使用中 (${free_status} 可用)                          ║"
      elif [[ "$free_count" -ge 4 ]]; then
        echo "║  └─ 狀態: ✅ ${free_status} GPU 可用                                   ║"
      else
        echo "║  └─ 狀態: ⚠️  ${free_status} GPU 可用                                   ║"
      fi
      echo "╠══════════════════════════════════════════════════════════════════╣"
      # 顯示每個 GPU 狀態
      while IFS=',' read -r idx util; do
        [[ -z "$idx" ]] && continue
        idx="${idx// /}"
        util="${util//[^0-9]/}"
        if [[ "$util" -lt 10 ]]; then
          printf "║  GPU %s: ✅ 閒置 (%3s%%)                                          ║\n" "$idx" "$util"
        else
          printf "║  GPU %s: 🔥 使用中 (%3s%%)                                        ║\n" "$idx" "$util"
        fi
      done <<< "$gpu_output"
    else
      echo "║  └─ 狀態: 🔴 無法取得 GPU 資訊                                   ║"
    fi
  else
    # 跳板節點 (如 .87 ib3)
    echo "║  📍 .${server} → ib${node}                                              ║"
    echo "║  ├─ 跳板: ${host}                                          ║"
    
    # 根據節點判斷 GPU 類型
    if [[ "$server" == "87" && "$node" == "6" ]]; then
      echo "║  ├─ GPU: 8× Tesla V100-SXM2-16GB ⚡                               ║"
    else
      echo "║  ├─ GPU: 8× Tesla P100-PCIE-16GB                                 ║"
    fi
    
    if gpu_output="$(ssh_batch_exec "$host" "ssh -o ConnectTimeout=3 cfdlab-ib${node} 'nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader'" 2>/dev/null)"; then
      free_status=$(get_gpu_status_line "$gpu_output")
      local free_count="${free_status%%/*}"
      if [[ "$free_count" -eq 0 ]]; then
        echo "║  └─ 狀態: ❌ 全部使用中 (${free_status} 可用)                          ║"
      elif [[ "$free_count" -ge 4 ]]; then
        echo "║  └─ 狀態: ✅ ${free_status} GPU 可用                                   ║"
      else
        echo "║  └─ 狀態: ⚠️  ${free_status} GPU 可用                                   ║"
      fi
      echo "╠══════════════════════════════════════════════════════════════════╣"
      # 顯示每個 GPU 狀態
      while IFS=',' read -r idx util; do
        [[ -z "$idx" ]] && continue
        idx="${idx// /}"
        util="${util//[^0-9]/}"
        if [[ "$util" -lt 10 ]]; then
          printf "║  GPU %s: ✅ 閒置 (%3s%%)                                          ║\n" "$idx" "$util"
        else
          printf "║  GPU %s: 🔥 使用中 (%3s%%)                                        ║\n" "$idx" "$util"
        fi
      done <<< "$gpu_output"
    else
      echo "║  └─ 狀態: 🔴 節點離線/維修中                                     ║"
      echo "╚══════════════════════════════════════════════════════════════════╝"
      echo ""
      die "ib${node} 無法連線，請選擇其他節點"
    fi
  fi
  
  echo "╚══════════════════════════════════════════════════════════════════╝"
  echo ""
  echo "🚀 正在連線..."
  echo ""

  # 確保遠端資料夾存在
  ensure_remote_dir "$server"

  # node=0 表示直連伺服器（例如 .89）
  if [[ "$node" == "0" ]]; then
    if [[ -n "$CFDLAB_PASSWORD" ]]; then
      sshpass -p "$CFDLAB_PASSWORD" ssh -t \
        -o StrictHostKeyChecking=accept-new \
        "${CFDLAB_USER}@${host}" \
        "cd ${CFDLAB_REMOTE_PATH}; exec bash"
    else
      ssh -t "${CFDLAB_USER}@${host}" "cd ${CFDLAB_REMOTE_PATH}; exec bash"
    fi
  else
    if [[ -n "$CFDLAB_PASSWORD" ]]; then
      # Use ProxyCommand with sshpass for both hops (local sshpass only, no sshpass needed on server)
      local proxy_cmd="sshpass -p '${CFDLAB_PASSWORD}' ssh -o StrictHostKeyChecking=accept-new -W %h:%p ${CFDLAB_USER}@${host}"
      sshpass -p "$CFDLAB_PASSWORD" ssh -t \
        -o StrictHostKeyChecking=accept-new \
        -o ProxyCommand="$proxy_cmd" \
        "${CFDLAB_USER}@cfdlab-ib${node}" \
        "cd ${CFDLAB_REMOTE_PATH}; exec bash"
    else
      ssh -t "${CFDLAB_USER}@${host}" "ssh -t cfdlab-ib${node} 'cd ${CFDLAB_REMOTE_PATH}; exec bash'"
    fi
  fi
}

function cmd_run() {
  local parsed
  local server
  local node
  local gpu_count
  local remote_cmd

  parsed="$(parse_combo "${1:-87:${CFDLAB_DEFAULT_NODE}}")"
  server="${parsed%%:*}"
  node="${parsed##*:}"
  gpu_count="${2:-$CFDLAB_DEFAULT_GPU_COUNT}"
  [[ "$gpu_count" =~ ^[0-9]+$ ]] || die "gpu_count must be an integer"

  remote_cmd="cd ${CFDLAB_REMOTE_PATH} && nvcc main.cu -arch=${CFDLAB_NVCC_ARCH} -I${CFDLAB_MPI_INCLUDE} -L${CFDLAB_MPI_LIB} -lmpi -o a.out && nohup mpirun -np ${gpu_count} /usr/local/cuda-10.2/bin/nsys profile -t cuda,nvtx -o ${CFDLAB_REMOTE_PATH}/nsys_rank%q{OMPI_COMM_WORLD_RANK} --duration=600 -f true ./a.out > log\$(date +%Y%m%d) 2>&1 &"
  run_on_node "$server" "$node" "$remote_cmd"
}

# ---- 拆分版：只編譯 ----
function cmd_compile() {
  local parsed
  local server
  local node
  local remote_cmd

  parsed="$(parse_combo "${1:-87:${CFDLAB_DEFAULT_NODE}}")"
  server="${parsed%%:*}"
  node="${parsed##*:}"

  remote_cmd="cd ${CFDLAB_REMOTE_PATH} && nvcc main.cu -arch=${CFDLAB_NVCC_ARCH} -I${CFDLAB_MPI_INCLUDE} -L${CFDLAB_MPI_LIB} -lmpi -o a.out"
  run_on_node "$server" "$node" "$remote_cmd"
}

# ---- 拆分版：只執行（需先編譯）----
function cmd_execute() {
  local parsed
  local server
  local node
  local gpu_count
  local remote_cmd

  parsed="$(parse_combo "${1:-87:${CFDLAB_DEFAULT_NODE}}")"
  server="${parsed%%:*}"
  node="${parsed##*:}"
  gpu_count="${2:-$CFDLAB_DEFAULT_GPU_COUNT}"
  [[ "$gpu_count" =~ ^[0-9]+$ ]] || die "gpu_count must be an integer"

  remote_cmd="cd ${CFDLAB_REMOTE_PATH} && nohup mpirun -np ${gpu_count} /usr/local/cuda-10.2/bin/nsys profile -t cuda,nvtx -o ${CFDLAB_REMOTE_PATH}/nsys_rank%q{OMPI_COMM_WORLD_RANK} --duration=600 -f true ./a.out > log\$(date +%Y%m%d) 2>&1 &"
  run_on_node "$server" "$node" "$remote_cmd"
}

function cmd_jobs() {
  local parsed
  local server
  local node

  parsed="$(parse_combo "${1:-87:${CFDLAB_DEFAULT_NODE}}")"
  server="${parsed%%:*}"
  node="${parsed##*:}"
  run_on_node "$server" "$node" "ps aux | grep a.out | grep -v grep || true"
}

function cmd_kill() {
  local parsed
  local server
  local node

  parsed="$(parse_combo "${1:-87:${CFDLAB_DEFAULT_NODE}}")"
  server="${parsed%%:*}"
  node="${parsed##*:}"
  run_on_node "$server" "$node" "pkill -f a.out || pkill -f mpirun || true"
}

# GPU 狀態查詢功能
function get_gpu_status_line() {
  # 解析 nvidia-smi 輸出，計算可用 GPU 數量
  local output="$1"
  local total=0
  local free=0
  local busy=0
  
  while IFS=',' read -r idx util rest; do
    [[ -z "$idx" ]] && continue
    util="${util//[^0-9]/}"  # 只保留數字
    [[ -z "$util" ]] && continue
    ((total++))
    if [[ "$util" -lt 10 ]]; then
      ((free++))
    else
      ((busy++))
    fi
  done <<< "$output"
  
  if [[ "$total" -eq 0 ]]; then
    echo "0/0"
  else
    echo "${free}/${total}"
  fi
}

function cmd_gpus() {
  echo ""
  echo "╔══════════════════════════════════════════════════════════════════╗"
  echo "║                    🖥️  GPU 狀態總覽                              ║"
  echo "╠══════════════════════════════════════════════════════════════════╣"
  
  # === .89 (直連，8x V100-32GB) ===
  echo "║                                                                  ║"
  echo "║  📍 .89 (140.114.58.89) - 直連伺服器                             ║"
  echo "║  ├─ GPU: 8× Tesla V100-SXM2-32GB                                 ║"
  
  local gpu89_output
  if gpu89_output="$(ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new "${CFDLAB_USER}@140.114.58.89" "nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader" 2>/dev/null)"; then
    local free89=$(get_gpu_status_line "$gpu89_output")
    local free_count="${free89%%/*}"
    if [[ "$free_count" -eq 0 ]]; then
      echo "║  └─ 狀態: ❌ 全部使用中 (${free89} 可用)                          ║"
    elif [[ "$free_count" -ge 4 ]]; then
      echo "║  └─ 狀態: ✅ ${free89} GPU 可用                                   ║"
    else
      echo "║  └─ 狀態: ⚠️  ${free89} GPU 可用                                   ║"
    fi
    # 顯示每個 GPU 狀態
    while IFS=',' read -r idx util; do
      [[ -z "$idx" ]] && continue
      idx="${idx// /}"
      util="${util//[^0-9]/}"
      if [[ "$util" -lt 10 ]]; then
        printf "║     GPU %s: ✅ 閒置 (%s%%)                                       ║\n" "$idx" "$util"
      else
        printf "║     GPU %s: 🔥 使用中 (%s%%)                                     ║\n" "$idx" "$util"
      fi
    done <<< "$gpu89_output"
  else
    echo "║  └─ 狀態: 🔴 無法連線                                          ║"
  fi
  
  echo "╠══════════════════════════════════════════════════════════════════╣"
  
  # === .87 (ib2, ib3, ib5, ib6) ===
  echo "║                                                                  ║"
  echo "║  📍 .87 (140.114.58.87) - 跳板伺服器                             ║"
  echo "║  ├─ GPU: 8× Tesla P100-PCIE-16GB (每節點)                        ║"
  
  local nodes_87="2 3 5 6"
  for node in $nodes_87; do
    local node_output
    if node_output="$(ssh_batch_exec "140.114.58.87" "ssh -o ConnectTimeout=3 cfdlab-ib${node} 'nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader'" 2>/dev/null)"; then
      local free_status=$(get_gpu_status_line "$node_output")
      local free_count="${free_status%%/*}"
      if [[ "$free_count" -eq 0 ]]; then
        printf "║  ├─ ib%s: ❌ 全部使用中 (%s 可用)                               ║\n" "$node" "$free_status"
      elif [[ "$free_count" -ge 4 ]]; then
        printf "║  ├─ ib%s: ✅ %s GPU 可用                                       ║\n" "$node" "$free_status"
      else
        printf "║  ├─ ib%s: ⚠️  %s GPU 可用                                       ║\n" "$node" "$free_status"
      fi
    else
      printf "║  ├─ ib%s: 🔴 維修中/無法連線                                    ║\n" "$node"
    fi
  done
  
  echo "╠══════════════════════════════════════════════════════════════════╣"
  
  # === .154 (ib1, ib4, ib7, ib9) ===
  echo "║                                                                  ║"
  echo "║  📍 .154 (140.114.58.154) - 跳板伺服器                           ║"
  echo "║  ├─ GPU: 8× Tesla P100-PCIE-16GB (每節點)                        ║"
  
  local nodes_154="1 4 7 9"
  for node in $nodes_154; do
    local node_output
    if node_output="$(ssh_batch_exec "140.114.58.154" "ssh -o ConnectTimeout=3 cfdlab-ib${node} 'nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader'" 2>/dev/null)"; then
      local free_status=$(get_gpu_status_line "$node_output")
      local free_count="${free_status%%/*}"
      if [[ "$free_count" -eq 0 ]]; then
        printf "║  ├─ ib%s: ❌ 全部使用中 (%s 可用)                               ║\n" "$node" "$free_status"
      elif [[ "$free_count" -ge 4 ]]; then
        printf "║  ├─ ib%s: ✅ %s GPU 可用                                       ║\n" "$node" "$free_status"
      else
        printf "║  ├─ ib%s: ⚠️  %s GPU 可用                                       ║\n" "$node" "$free_status"
      fi
    else
      printf "║  ├─ ib%s: 🔴 維修中/無法連線                                    ║\n" "$node"
    fi
  done
  
  echo "╠══════════════════════════════════════════════════════════════════╣"
  echo "║  💡 說明: ✅ 可用 | ⚠️ 部分可用 | ❌ 全滿 | 🔴 離線/維修        ║"
  echo "║  📝 可用 = GPU 使用率 < 10%                                      ║"
  echo "╚══════════════════════════════════════════════════════════════════╝"
  echo ""
}

function cmd_gpu_detail() {
  local target="${1:-all}"
  
  echo ""
  echo "=== GPU 詳細狀態 ==="
  echo ""
  
  case "$target" in
    89|.89)
      echo "📍 .89 (140.114.58.89) - 8× Tesla V100-SXM2-32GB"
      echo "─────────────────────────────────────────────────"
      ssh -o ConnectTimeout=8 "${CFDLAB_USER}@140.114.58.89" "nvidia-smi" 2>/dev/null || echo "❌ 無法連線"
      ;;
    87|.87)
      echo "📍 .87 節點狀態"
      for node in 2 3 5 6; do
        echo ""
        echo "=== .87 ib${node} ==="
        ssh_batch_exec "140.114.58.87" "ssh -o ConnectTimeout=3 cfdlab-ib${node} 'nvidia-smi'" 2>/dev/null || echo "❌ ib${node} 無法連線/維修中"
      done
      ;;
    154|.154)
      echo "📍 .154 節點狀態"
      for node in 1 4 7 9; do
        echo ""
        echo "=== .154 ib${node} ==="
        ssh_batch_exec "140.114.58.154" "ssh -o ConnectTimeout=3 cfdlab-ib${node} 'nvidia-smi'" 2>/dev/null || echo "❌ ib${node} 無法連線/維修中"
      done
      ;;
    all|*)
      cmd_gpu_detail 89
      echo ""
      cmd_gpu_detail 87
      echo ""
      cmd_gpu_detail 154
      ;;
  esac
}

function cmd_log() {
  local target
  local lines

  target="$(parse_server_or_all "${1:-all}")"
  lines="${2:-20}"
  [[ "$lines" =~ ^[0-9]+$ ]] || die "tail_lines must be an integer"

  while IFS= read -r server; do
    [[ -z "$server" ]] && continue
    echo "=== ${server} remote logs ==="
    run_on_server "$server" "ls -lth ${CFDLAB_REMOTE_PATH}/log* 2>/dev/null | head -10 || true"
    echo
    echo "=== latest log tail (${lines}) ==="
    run_on_server "$server" "latest=\$(ls -t ${CFDLAB_REMOTE_PATH}/log* 2>/dev/null | head -1); [[ -n \"\$latest\" ]] && tail -n ${lines} \"\$latest\" || echo 'No log files found'"
    echo
  done < <(each_target_server "$target")
}

function cmd_pull_like() {
  local mode="$1"
  shift
  local diff_mode="full"
  local confirm="false"
  local positional=()

  # Parse options
  for arg in "$@"; do
    case "$arg" in
      --no-diff)       diff_mode="no-diff" ;;
      --diff-summary)  diff_mode="summary" ;;
      --diff-stat)     diff_mode="stat" ;;
      --diff-full)     diff_mode="full" ;;
      --force)         confirm="false"; diff_mode="no-diff" ;;
      --quick)         diff_mode="no-diff" ;;
      *)               positional+=("$arg") ;;
    esac
  done

  local server
  server="$(normalize_server "${positional[0]:-87}")"
  local delete_mode
  if [[ "$mode" == "pull" ]]; then
    delete_mode="delete"
  else
    delete_mode="keep"
  fi

  # Show diff analysis before transfer (manual commands only)
  if [[ "$diff_mode" != "no-diff" ]]; then
    if ! analyze_code_diff "$mode" "$server" "$diff_mode" "$confirm"; then
      return 1
    fi
  fi

  multi_server_run "$mode" "$server" "$delete_mode"

  # Log to history
  log_sync_history "$(echo "$mode" | tr '[:lower:]' '[:upper:]')" "$server" "${DIFF_FILE_COUNT:-0}" "${DIFF_TOTAL_ADDS:-0}" "${DIFF_TOTAL_DELS:-0}"
}

function cmd_push_like() {
  local mode="$1"
  shift
  local diff_mode="full"
  local confirm="true"  # push defaults to confirm
  local positional=()

  # Parse options
  for arg in "$@"; do
    case "$arg" in
      --no-diff)       diff_mode="no-diff"; confirm="false" ;;
      --diff-summary)  diff_mode="summary" ;;
      --diff-stat)     diff_mode="stat" ;;
      --diff-full)     diff_mode="full" ;;
      --force)         confirm="false"; diff_mode="no-diff" ;;
      --quick)         diff_mode="no-diff"; confirm="false" ;;
      *)               positional+=("$arg") ;;
    esac
  done

  local target
  target="$(parse_server_or_all "${positional[0]:-all}")"
  local delete_mode="delete"
  if [[ "$mode" == "push-keep" ]]; then
    delete_mode="keep"
  fi

  # Show diff analysis before transfer
  if [[ "$diff_mode" != "no-diff" ]]; then
    # For push, show diff on the first target server
    local first_server
    first_server="$(each_target_server "$target" | head -1)"
    if ! analyze_code_diff push "$first_server" "$diff_mode" "$confirm"; then
      return 1
    fi
  fi

  multi_server_run push "$target" "$delete_mode"

  # Log to history
  local first_server
  first_server="$(each_target_server "$target" | head -1)"
  log_sync_history PUSH "$first_server" "${DIFF_FILE_COUNT:-0}" "${DIFF_TOTAL_ADDS:-0}" "${DIFF_TOTAL_DELS:-0}"
}

function cmd_diff() {
  local diff_mode="full"
  local positional=()

  for arg in "$@"; do
    case "$arg" in
      --summary)  diff_mode="summary" ;;
      --stat)     diff_mode="stat" ;;
      --full)     diff_mode="full" ;;
      *)          positional+=("$arg") ;;
    esac
  done

  local target
  target="$(parse_server_or_all "${positional[0]:-all}")"

  local servers=()
  local total=0
  while IFS= read -r s; do
    [[ -z "$s" ]] && continue
    servers+=("$s")
    ((total++))
  done < <(each_target_server "$target")

  local idx=0
  for server in "${servers[@]}"; do
    ((idx++))
    local color emoji host
    color="$(server_color "$server")"
    emoji="$(server_emoji "$server")"
    host="$(resolve_host "$server")"

    printf '%b══════════════════════════════════════════════════%b\n' "$color" "$CLR_RST"
    printf '%b [%d/%d] %s Diff .%s (%s)%b\n' "$color" "$idx" "$total" "$emoji" "$server" "$host" "$CLR_RST"
    printf '%b══════════════════════════════════════════════════%b\n' "$color" "$CLR_RST"

    echo ""
    printf '%b--- Push diff (local → remote) ---%b\n' "$CLR_BOLD" "$CLR_RST"
    analyze_code_diff push "$server" "$diff_mode" false

    echo ""
    printf '%b--- Pull diff (remote → local) ---%b\n' "$CLR_BOLD" "$CLR_RST"
    analyze_code_diff pull "$server" "$diff_mode" false
    echo ""
  done
}

function cmd_add() {
  local target
  local combined=""

  target="$(parse_server_or_all "${1:-all}")"
  while IFS= read -r server; do
    [[ -z "$server" ]] && continue
    local preview
    if ! preview="$(preview_push_changes "$server")"; then
      echo "[ERROR] add preview failed for ${server}"
      continue
    fi
    combined+="$preview"
    combined+=$'\n'
  done < <(each_target_server "$target")

  local paths
  paths="$(list_change_paths "$combined" | LC_ALL=C sort -u)"
  if [[ -z "$paths" ]]; then
    echo "No pending source changes"
    return
  fi

  echo "Files to be pushed:"
  printf '%s\n' "$paths"
}

function cmd_status() {
  local target
  target="$(parse_server_or_all "${1:-all}")"

  while IFS= read -r server; do
    [[ -z "$server" ]] && continue
    print_pending_summary "$server"
  done < <(each_target_server "$target")
}

function cmd_syncstatus() {
  local target
  target="$(parse_server_or_all "${1:-all}")"

  echo "=== Pending Sync ==="
  cmd_status "$target"
  echo
  echo "=== Watch Daemons ==="
  status_daemon push
  status_daemon pull
}

function cmd_issynced() {
  local out=()
  local server

  for server in 87 89 154; do
    local push_preview
    local pull_preview
    local push_count
    local pull_count

    if ! push_preview="$(preview_push_changes "$server")"; then
      out+=(".${server}: [ERR]")
      continue
    fi
    if ! pull_preview="$(preview_pull_changes "$server" keep)"; then
      out+=(".${server}: [ERR]")
      continue
    fi
    push_count="$(count_change_lines "$push_preview")"
    pull_count="$(count_change_lines "$pull_preview")"

    if [[ "$push_count" -eq 0 && "$pull_count" -eq 0 ]]; then
      out+=(".${server}: [OK]")
    else
      out+=(".${server}: [DIFF]")
    fi
  done

  printf '%s | %s | %s\n' "${out[0]}" "${out[1]}" "${out[2]}"
}

function cmd_autopull() {
  local target
  target="$(parse_server_or_all "${1:-all}")"

  while IFS= read -r server; do
    [[ -z "$server" ]] && continue
    local preview
    local count

    if ! preview="$(preview_pull_changes "$server" delete)"; then
      echo "[AUTOPULL] ${server}: preview failed"
      continue
    fi
    count="$(count_change_lines "$preview")"

    if [[ "$count" -gt 0 ]]; then
      echo "[AUTOPULL] ${server}: ${count} changes"
      git_style_transfer pull "$server" delete
    else
      echo "[AUTOPULL] ${server}: no changes"
    fi
  done < <(each_target_server "$target")
}

function cmd_autofetch() {
  local target
  target="$(parse_server_or_all "${1:-all}")"

  while IFS= read -r server; do
    [[ -z "$server" ]] && continue
    local preview
    local count

    if ! preview="$(preview_pull_changes "$server" keep)"; then
      echo "[AUTOFETCH] ${server}: preview failed"
      continue
    fi
    count="$(count_change_lines "$preview")"

    if [[ "$count" -gt 0 ]]; then
      echo "[AUTOFETCH] ${server}: ${count} changes"
      git_style_transfer fetch "$server" keep
    else
      echo "[AUTOFETCH] ${server}: no changes"
    fi
  done < <(each_target_server "$target")
}

function cmd_autopush() {
  local target
  target="$(parse_server_or_all "${1:-all}")"

  while IFS= read -r server; do
    [[ -z "$server" ]] && continue
    local preview
    local count

    if ! preview="$(preview_push_changes "$server")"; then
      echo "[AUTOPUSH] ${server}: preview failed"
      continue
    fi
    count="$(count_change_lines "$preview")"

    if [[ "$count" -gt 0 ]]; then
      echo "[AUTOPUSH] ${server}: ${count} changes"
      git_style_transfer push "$server" delete
    else
      echo "[AUTOPUSH] ${server}: no changes"
    fi
  done < <(each_target_server "$target")
}

function cmd_reset() {
  local target
  target="$(parse_server_or_all "${1:-all}")"
  confirm_or_die "Delete remote-only source files for target=${target}?"

  while IFS= read -r server; do
    [[ -z "$server" ]] && continue
    local host
    local args=()

    host="$(resolve_host "$server")"
    while IFS= read -r line; do
      args+=("$line")
    done < <(build_arg_array push delete)

    echo "[RESET] remote cleanup on ${server}"
    rsync "${args[@]}" --existing --ignore-existing "${WORKSPACE_DIR}/" "${CFDLAB_USER}@${host}:${CFDLAB_REMOTE_PATH}/"
  done < <(each_target_server "$target")
}

function cmd_clone() {
  local server
  local host

  server="$(normalize_server "${1:-87}")"
  host="$(resolve_host "$server")"

  confirm_or_die "Clone from ${server} and overwrite local files?"

  rsync -az --delete --exclude=.git/ --exclude=.vscode/ -e ssh \
    "${CFDLAB_USER}@${host}:${CFDLAB_REMOTE_PATH}/" \
    "${WORKSPACE_DIR}/"
}

# ========== New Diff / Sync Commands ==========

function cmd_sync_diff() {
  local diff_mode="full"
  local positional=()

  for arg in "$@"; do
    case "$arg" in
      --summary) diff_mode="summary" ;;
      --stat)    diff_mode="stat" ;;
      --full)    diff_mode="full" ;;
      *)         positional+=("$arg") ;;
    esac
  done

  local server
  server="$(normalize_server "${positional[0]:-87}")"
  printf '\033[1m🔄 Comparing local ↔ .%s (no transfer)\033[0m\n' "$server"

  echo ""
  printf '\033[1m=== Push direction (local → remote) ===\033[0m\n'
  analyze_code_diff push "$server" "$diff_mode" false

  echo ""
  printf '\033[1m=== Pull direction (remote → local) ===\033[0m\n'
  analyze_code_diff pull "$server" "$diff_mode" false
}

function cmd_sync_diff_file() {
  local filename="${1:?Usage: sync-diff-file <filename> [server]}"
  local server
  server="$(normalize_server "${2:-87}")"

  printf '\033[1m📄 Diff for: %s (vs .%s)\033[0m\n' "$filename" "$server"

  local local_file="${WORKSPACE_DIR}/${filename}"
  if [[ ! -f "$local_file" ]]; then
    printf '\033[31m[ERROR] Local file not found: %s\033[0m\n' "$filename"
    return 1
  fi

  DIFF_TOTAL_ADDS=0
  DIFF_TOTAL_DELS=0
  show_file_diff push "$server" "$filename" "modified" "full"
}

function cmd_sync_log() {
  local lines="${1:-30}"
  if [[ ! -f "$SYNC_HISTORY_FILE" ]]; then
    echo "No sync history yet."
    return
  fi
  printf '\033[1m📝 Sync History (last %d entries):\033[0m\n' "$lines"
  tail -n "$lines" "$SYNC_HISTORY_FILE"
}

function cmd_sync_stop() {
  echo "Stopping all background sync daemons..."
  stop_daemon push 2>/dev/null || true
  stop_daemon pull 2>/dev/null || true
  stop_daemon fetch 2>/dev/null || true
  echo "All sync daemons stopped."
}

# ========== End New Diff / Sync Commands ==========

function cmd_sync() {
  cmd_diff all
  confirm_or_die "Push local source changes to both servers?"
  cmd_push_like push all --no-diff
}

function cmd_fullsync() {
  confirm_or_die "Push + delete remote-only source files on both servers?"
  cmd_push_like push-sync all --no-diff
}

function cmd_watch_generic() {
  local kind="$1"
  shift || true

  local sub="${1:-start}"
  case "$sub" in
    status)
      status_daemon "$kind"
      ;;
    log)
      show_daemon_log "$kind"
      ;;
    stop)
      stop_daemon "$kind"
      ;;
    clear)
      clear_daemon_log "$kind"
      ;;
    *)
      local target interval
      if [[ "$kind" == "push" ]]; then
        target="$(parse_server_or_all "${1:-all}")"
        interval="${2:-10}"
      elif [[ "$kind" == "vtkrename" ]]; then
        target="local"
        interval="${1:-5}"
      else
        target="$(parse_server_or_all "${1:-all}")"
        interval="${2:-30}"
      fi
      start_daemon "$kind" "$target" "$interval"
      ;;
  esac
}

function cmd_watch() {
  cmd_watch_generic push "$@"
}

function cmd_watchpush() {
  cmd_watch_generic push "$@"
}

function cmd_watchpull() {
  cmd_watch_generic pull "$@"
}

function cmd_watchfetch() {
  cmd_watch_generic fetch "$@"
}

function cmd_vtkrename() {
  cmd_watch_generic vtkrename "$@"
}

function cmd_bgstatus() {
  status_daemon push
  status_daemon pull
  status_daemon fetch
  status_daemon vtkrename
}

function cmd_check() {
  local server

  require_cmd ssh
  require_cmd rsync
  ensure_password_tooling
  echo "[CHECK] local commands OK (ssh, rsync)"

  for server in 87 154; do
    local host
    host="$(resolve_host "$server")"
    if ssh_batch_exec "$host" "echo ok" >/dev/null 2>&1; then
      echo "[CHECK] ssh ${server} (${host}) OK"
    else
      echo "[CHECK] ssh ${server} (${host}) FAILED"
    fi
  done
}

function main() {
  local cmd="${1:-help}"
  shift || true

  if [[ "$cmd" == "help" || "$cmd" == "-h" || "$cmd" == "--help" ]]; then
    cmd_help
    return
  fi

  if [[ "$cmd" == "__daemon_loop" ]]; then
    daemon_loop "$1" "$2" "$3"
    return
  fi

  require_cmd ssh
  ensure_password_tooling

  # Auto-fix VPN route before any remote operation
  case "$cmd" in
    help|-h|--help|bgstatus|vtkrename) ;; # local-only, skip
    *) ensure_vpn_route ;;
  esac

  case "$cmd" in
    add|autofetch|autofetch87|autofetch89|autofetch154|autopull|autopull87|autopull89|autopull154|autopush|autopush87|autopush89|autopush154|autopushall|bgstatus|check|clone|delete|diff|diff87|diff89|diff154|diffall|fetch|fetch154|fetch87|fetch89|fullsync|issynced|log|log87|log89|log154|pull|pull154|pull87|pull89|push|push87|push89|push154|pushall|reset|status|sync|sync-diff|sync-diff-summary|sync-diff-file|sync-log|sync-stop|syncstatus|vtkrename|watch|watchfetch|watchpull|watchpush)
      require_cmd rsync
      ;;
  esac

  case "$cmd" in
    add) cmd_add "$@" ;;
    autofetch) cmd_autofetch "$@" ;;
    autofetch87) cmd_autofetch 87 ;;
    autofetch89) cmd_autofetch 89 ;;
    autofetch154) cmd_autofetch 154 ;;
    autopull) cmd_autopull "$@" ;;
    autopull87) cmd_autopull 87 ;;
    autopull89) cmd_autopull 89 ;;
    autopull154) cmd_autopull 154 ;;
    autopush) cmd_autopush "$@" ;;
    autopush87) cmd_autopush 87 ;;
    autopush89) cmd_autopush 89 ;;
    autopush154) cmd_autopush 154 ;;
    autopushall) cmd_autopush all ;;
    bgstatus) cmd_bgstatus ;;
    check) cmd_check ;;
    clone) cmd_clone "$@" ;;
    delete) cmd_reset "$@" ;;
    diff) cmd_diff "$@" ;;
    diff87) cmd_diff 87 ;;
    diff89) cmd_diff 89 ;;
    diff154) cmd_diff 154 ;;
    diffall) cmd_diff all ;;
    fetch) cmd_pull_like fetch "$@" ;;
    fetch87) cmd_pull_like fetch 87 ;;
    fetch89) cmd_pull_like fetch 89 ;;
    fetch154) cmd_pull_like fetch 154 ;;
    fullsync) cmd_fullsync ;;
    issynced) cmd_issynced ;;
    log) cmd_log "$@" ;;
    log87) cmd_log 87 ;;
    log89) cmd_log 89 ;;
    log154) cmd_log 154 ;;
    pull) cmd_pull_like pull "$@" ;;
    pull87) cmd_pull_like pull 87 ;;
    pull89) cmd_pull_like pull 89 ;;
    pull154) cmd_pull_like pull 154 ;;
    push) cmd_push_like push "$@" ;;
    push87) cmd_push_like push 87 "$@" ;;
    push89) cmd_push_like push 89 "$@" ;;
    push154) cmd_push_like push 154 "$@" ;;
    pushall) cmd_push_like push all "$@" ;;
    reset) cmd_reset "$@" ;;
    status) cmd_status "$@" ;;
    sync) cmd_sync ;;
    sync-diff) cmd_sync_diff "$@" ;;
    sync-diff-summary) cmd_sync_diff --summary "$@" ;;
    sync-diff-file) cmd_sync_diff_file "$@" ;;
    sync-log) cmd_sync_log "$@" ;;
    sync-stop) cmd_sync_stop ;;
    syncstatus) cmd_syncstatus "$@" ;;
    vtkrename) cmd_vtkrename "$@" ;;
    watch) cmd_watch "$@" ;;
    watchfetch) cmd_watchfetch "$@" ;;
    watchpull) cmd_watchpull "$@" ;;
    watchpush) cmd_watchpush "$@" ;;

    ssh) cmd_ssh "$@" ;;
    issh) cmd_issh "$@" ;;
    issh-quick) cmd_issh_quick "$@" ;;
    run) cmd_run "$@" ;;
    compile) cmd_compile "$@" ;;
    execute) cmd_execute "$@" ;;
    jobs) cmd_jobs "$@" ;;
    kill) cmd_kill "$@" ;;
    gpus) cmd_gpus ;;
    gpu) cmd_gpu_detail "$@" ;;
    vpnfix) vpn_fix_route ;;

    *)
      cmd_help
      die "Unknown command: $cmd"
      ;;
  esac
}

main "$@"
