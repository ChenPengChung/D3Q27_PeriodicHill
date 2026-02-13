#!/usr/bin/env bash
set -euo pipefail

# Ensure Homebrew paths are available (macOS arm64 & intel)
[[ -d /opt/homebrew/bin ]] && export PATH="/opt/homebrew/bin:$PATH"
[[ -d /usr/local/bin ]]    && export PATH="/usr/local/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
    --exclude=.git/
    --exclude=.vscode/
    --exclude=backup/
    --exclude=result/
    --exclude=statistics/
    --exclude=a.out
    --exclude=*.o
    --exclude=*.exe
    --exclude=*.dat
    --exclude=*.DAT
    --exclude=*.plt
    --exclude=*.bin
    --exclude=*.vtk
    --exclude=log*
    -e
    "$rsh_cmd"
  )

  if [[ "$delete_mode" == "delete" ]]; then
    args+=(--delete)
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
  printf '%s\n' "$output"
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
  printf '%s\n' "$output"
}

function count_change_lines() {
  local payload="$1"
  printf '%s\n' "$payload" | awk 'NF > 0 && $1 ~ /^[<>ch\.\*]/ {c++} END{print c+0}'
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

Core (same names as Windows):
  diff, check, status, add, push, pull, fetch, log
  reset, delete, clone, sync, fullsync, issynced
  autopush, autopull, autofetch
  watch, watchpush, watchpull, watchfetch
  syncstatus, bgstatus, vtkrename
  pull87, pull89, pull154, fetch87, fetch89, fetch154
  push87, push89, push154, pushall
  autopull87, autopull89, autopull154
  autofetch87, autofetch89, autofetch154
  autopush87, autopush89, autopush154, autopushall
  diff87, diff89, diff154, diffall
  log87, log89, log154

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
  ./.vscode/cfdlab-mac.sh diff
  ./.vscode/cfdlab-mac.sh push
  ./.vscode/cfdlab-mac.sh watchpush all 10
  ./.vscode/cfdlab-mac.sh watchpull status
  ./.vscode/cfdlab-mac.sh vtkrename
EOF
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

  remote_cmd="cd ${CFDLAB_REMOTE_PATH} && nvcc main.cu -arch=${CFDLAB_NVCC_ARCH} -I${CFDLAB_MPI_INCLUDE} -L${CFDLAB_MPI_LIB} -lmpi -o a.out && nohup mpirun -np ${gpu_count} ./a.out > log\$(date +%Y%m%d) 2>&1 &"
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
  local server
  local lines

  server="$(normalize_server "${1:-87}")"
  lines="${2:-20}"
  [[ "$lines" =~ ^[0-9]+$ ]] || die "tail_lines must be an integer"

  echo "=== ${server} remote logs ==="
  run_on_server "$server" "ls -lth ${CFDLAB_REMOTE_PATH}/log* 2>/dev/null | head -10 || true"
  echo
  echo "=== latest log tail (${lines}) ==="
  run_on_server "$server" "latest=\$(ls -t ${CFDLAB_REMOTE_PATH}/log* 2>/dev/null | head -1); [[ -n \"\$latest\" ]] && tail -n ${lines} \"\$latest\" || echo 'No log files found'"
}

function cmd_pull_like() {
  local mode="$1"
  local server

  server="$(normalize_server "${2:-87}")"
  echo "[SYNC] ${mode} from ${server}"

  if [[ "$mode" == "fetch" ]]; then
    run_pull "$server" delete
  else
    run_pull "$server" keep
  fi
}

function cmd_push_like() {
  local mode="$1"
  local target
  local delete_mode

  target="$(parse_server_or_all "${2:-all}")"
  # Match Windows behavior: push also deletes remote-only source files
  # (rsync exclusions ensure result/log/data files are never deleted)
  delete_mode="delete"
  if [[ "$mode" == "push-keep" ]]; then
    delete_mode="keep"
  fi

  while IFS= read -r server; do
    [[ -z "$server" ]] && continue
    echo "[SYNC] ${mode} to ${server}"
    run_push "$server" "$delete_mode"
  done < <(each_target_server "$target")
}

function cmd_diff() {
  local target
  target="$(parse_server_or_all "${1:-all}")"

  while IFS= read -r server; do
    [[ -z "$server" ]] && continue
    echo "=== ${server} push preview ==="
    preview_push_changes "$server" || echo "[ERROR] push preview failed for ${server}"
    echo "=== ${server} pull preview ==="
    preview_pull_changes "$server" keep || echo "[ERROR] pull preview failed for ${server}"
  done < <(each_target_server "$target")
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
  paths="$(list_change_paths "$combined" | sort -u)"
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
  local server
  local preview
  local count

  server="$(normalize_server "${1:-87}")"
  if ! preview="$(preview_pull_changes "$server" keep)"; then
    echo "[AUTOPULL] ${server}: preview failed"
    return 1
  fi
  count="$(count_change_lines "$preview")"

  if [[ "$count" -gt 0 ]]; then
    echo "[AUTOPULL] ${server}: ${count} changes"
    run_pull "$server" keep
  else
    echo "[AUTOPULL] ${server}: no changes"
  fi
}

function cmd_autofetch() {
  local server
  local preview
  local count

  server="$(normalize_server "${1:-87}")"
  if ! preview="$(preview_pull_changes "$server" delete)"; then
    echo "[AUTOFETCH] ${server}: preview failed"
    return 1
  fi
  count="$(count_change_lines "$preview")"

  if [[ "$count" -gt 0 ]]; then
    echo "[AUTOFETCH] ${server}: ${count} changes"
    run_pull "$server" delete
  else
    echo "[AUTOFETCH] ${server}: no changes"
  fi
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
      run_push "$server" delete
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

function cmd_sync() {
  cmd_diff all
  confirm_or_die "Push local source changes to both servers?"
  cmd_push_like push all
}

function cmd_fullsync() {
  confirm_or_die "Push + delete remote-only source files on both servers?"
  cmd_push_like push-sync all
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
        target="$(normalize_server "${1:-87}")"
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
  case "$cmd" in
    add|autofetch|autofetch87|autofetch89|autofetch154|autopull|autopull87|autopull89|autopull154|autopush|autopush87|autopush89|autopush154|autopushall|bgstatus|check|clone|delete|diff|diff87|diff89|diff154|diffall|fetch|fetch154|fetch87|fetch89|fullsync|issynced|log|log87|log89|log154|pull|pull154|pull87|pull89|push|push87|push89|push154|pushall|reset|status|sync|syncstatus|vtkrename|watch|watchfetch|watchpull|watchpush)
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
    push87) cmd_push_like push 87 ;;
    push89) cmd_push_like push 89 ;;
    push154) cmd_push_like push 154 ;;
    pushall) cmd_push_like push all ;;
    reset) cmd_reset "$@" ;;
    status) cmd_status "$@" ;;
    sync) cmd_sync ;;
    syncstatus) cmd_syncstatus "$@" ;;
    vtkrename) cmd_vtkrename "$@" ;;
    watch) cmd_watch "$@" ;;
    watchfetch) cmd_watchfetch "$@" ;;
    watchpull) cmd_watchpull "$@" ;;
    watchpush) cmd_watchpush "$@" ;;

    ssh) cmd_ssh "$@" ;;
    run) cmd_run "$@" ;;
    jobs) cmd_jobs "$@" ;;
    kill) cmd_kill "$@" ;;
    gpus) cmd_gpus ;;
    gpu) cmd_gpu_detail "$@" ;;

    *)
      cmd_help
      die "Unknown command: $cmd"
      ;;
  esac
}

main "$@"
