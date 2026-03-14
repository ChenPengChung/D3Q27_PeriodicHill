#!/bin/bash
# ============================================================================
#  PeriodicHill D3Q27 GILBM — Remote Workflow Keybindings
# ============================================================================
#  Usage:  source this file in ~/.bashrc on each remote server (.87/.89/.154)
#          echo 'source ~/ph_keybindings.sh' >> ~/.bashrc
#
#  Principle: 所有指令都假設「本地與遠端同名資料夾」。
#             腳本自動偵測當前目錄作為工作目錄。
# ============================================================================

# ─── User Config ────────────────────────────────────────────────────────────
# 你本地 Mac 的 IP（用於 Alt+H 推送 VTK 回本地）
# 如果用 SSH reverse tunnel，改成 localhost + port
PH_LOCAL_USER="pengzhong"          # ← 改成你的 Mac 使用者名稱
PH_LOCAL_IP=""                     # ← 填你 Mac IP，空白則只印 scp 指令
PH_LOCAL_BASE="~/Desktop"          # ← 本地接收 VTK 的基底路徑

# Python 執行器（遠端）
PH_PYTHON="python3"

# VTK 下載數量上限（預設抓最新 5 個）
PH_VTK_COUNT=5

# ─── Color Helpers ──────────────────────────────────────────────────────────
_ph_red()    { echo -e "\033[1;31m$*\033[0m"; }
_ph_green()  { echo -e "\033[1;32m$*\033[0m"; }
_ph_yellow() { echo -e "\033[1;33m$*\033[0m"; }
_ph_cyan()   { echo -e "\033[1;36m$*\033[0m"; }
_ph_bold()   { echo -e "\033[1m$*\033[0m"; }

# ─── Core: Get project dir name (同名原則) ──────────────────────────────────
_ph_projname() {
    basename "$(pwd)"
}

# ─── Core: Find result directory ────────────────────────────────────────────
_ph_resultdir() {
    # 優先找 ./result，其次 ./Result，再找 ./output
    for d in result Result output Output results Results; do
        if [ -d "./$d" ]; then
            echo "./$d"
            return 0
        fi
    done
    _ph_red "[ERROR] 找不到 result/ 資料夾，請確認當前目錄"
    return 1
}

# ============================================================================
#  Alt+F  →  執行 5.SL_python.py （Streamline 流線圖）
# ============================================================================
_ph_run_SL() {
    echo ""
    _ph_cyan "━━━ [Alt+F] 5.SL_python.py ━━━  dir: $(pwd)"
    if [ -f "5.SL_python.py" ]; then
        $PH_PYTHON 5.SL_python.py
        _ph_green "✓ 5.SL_python.py 執行完成"
    else
        _ph_red "✗ 找不到 5.SL_python.py（當前目錄: $(pwd)）"
    fi
}

# ============================================================================
#  Alt+T  →  執行 4.Ma_U_Time.py （Ma / U 時序圖）
# ============================================================================
_ph_run_MaUTime() {
    echo ""
    _ph_cyan "━━━ [Alt+T] 4.Ma_U_Time.py ━━━  dir: $(pwd)"
    if [ -f "4.Ma_U_Time.py" ]; then
        $PH_PYTHON 4.Ma_U_Time.py
        _ph_green "✓ 4.Ma_U_Time.py 執行完成"
    else
        _ph_red "✗ 找不到 4.Ma_U_Time.py（當前目錄: $(pwd)）"
    fi
}

# ============================================================================
#  Alt+P  →  執行 2.Benchmark.py （ERCOFTAC 基準比對）
# ============================================================================
_ph_run_Benchmark() {
    echo ""
    _ph_cyan "━━━ [Alt+P] 2.Benchmark.py ━━━  dir: $(pwd)"
    if [ -f "2.Benchmark.py" ]; then
        $PH_PYTHON 2.Benchmark.py
        _ph_green "✓ 2.Benchmark.py 執行完成"
    else
        _ph_red "✗ 找不到 2.Benchmark.py（當前目錄: $(pwd)）"
    fi
}

# ============================================================================
#  Alt+H  →  下載最新 VTK 到本地（同名資料夾）
# ============================================================================
_ph_download_vtk() {
    echo ""
    _ph_cyan "━━━ [Alt+H] 下載最新 VTK ━━━"

    local rdir
    rdir=$(_ph_resultdir) || return 1
    local proj=$(_ph_projname)

    # 找最新的 VTK 檔案（按修改時間排序）
    local vtk_files
    vtk_files=$(find "$rdir" -maxdepth 2 -name "*.vtk" -o -name "*.vtr" -o -name "*.vts" -o -name "*.vtu" -o -name "*.pvd" 2>/dev/null \
                | xargs ls -t 2>/dev/null | head -n "$PH_VTK_COUNT")

    if [ -z "$vtk_files" ]; then
        _ph_red "✗ $rdir 中找不到任何 VTK 檔案"
        return 1
    fi

    _ph_yellow "找到最新 $PH_VTK_COUNT 個 VTK 檔案："
    echo "$vtk_files" | while read -r f; do
        ls -lh "$f" 2>/dev/null | awk '{printf "  %-50s %s %s %s\n", $NF, $6, $7, $8}'
    done

    if [ -n "$PH_LOCAL_IP" ]; then
        # 直接推送到本地 Mac
        local dest="${PH_LOCAL_USER}@${PH_LOCAL_IP}:${PH_LOCAL_BASE}/${proj}/result/"
        _ph_yellow "推送到 → $dest"
        echo "$vtk_files" | xargs -I{} scp -q {} "$dest" && \
            _ph_green "✓ 已推送 VTK 到本地" || \
            _ph_red "✗ scp 失敗，請檢查 SSH 反向連線"
    else
        # 沒設定 IP，印出指令讓使用者在本地貼上
        _ph_yellow "── 請在本地 Mac 終端機執行以下指令 ──"
        local server_ip=$(hostname -I 2>/dev/null | awk '{print $1}')
        [ -z "$server_ip" ] && server_ip="<SERVER_IP>"
        local remote_user=$(whoami)
        echo ""
        _ph_bold "scp ${remote_user}@${server_ip}:$(pwd)/${rdir}/\$(ls -t $(pwd)/${rdir}/*.vtk 2>/dev/null | head -1 | xargs basename) ~/${proj}/"
        echo ""
        _ph_yellow "或批量下載最新 ${PH_VTK_COUNT} 個："
        echo "$vtk_files" | while read -r f; do
            echo "scp ${remote_user}@${server_ip}:$(realpath "$f") ~/Desktop/${proj}/"
        done
    fi
}

# ============================================================================
#  Alt+R  →  即時監控模擬 log（tail checkrho / Ustar_Force_record）
# ============================================================================
_ph_tail_log() {
    echo ""
    _ph_cyan "━━━ [Alt+R] 即時監控 ━━━"

    # 同時監控 checkrho.dat 和 Ustar_Force_record.dat
    local files_found=()
    for f in checkrho.dat Ustar_Force_record.dat nohup.out slurm-*.out; do
        # expand glob
        for g in $f; do
            [ -f "$g" ] && files_found+=("$g")
        done
    done

    if [ ${#files_found[@]} -eq 0 ]; then
        _ph_red "✗ 找不到任何 log 檔案"
        return 1
    fi

    _ph_yellow "監控中（Ctrl+C 退出）："
    printf '  %s\n' "${files_found[@]}"
    echo ""
    tail -f "${files_found[@]}"
}

# ============================================================================
#  Alt+S  →  模擬狀態總覽（一鍵看全部關鍵資訊）
# ============================================================================
_ph_status() {
    echo ""
    _ph_cyan "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    _ph_cyan "  PeriodicHill 狀態總覽  |  $(hostname)  |  $(_ph_projname)"
    _ph_cyan "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # 1) GPU 狀態
    _ph_bold "[GPU]"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null | \
        awk -F', ' '{printf "  GPU%s %-20s  使用率:%3s%%  記憶體:%s/%sMB  溫度:%s°C\n",$1,$2,$3,$4,$5,$6}'
    echo ""

    # 2) 模擬程序是否在跑
    _ph_bold "[Process]"
    local procs=$(ps aux | grep -E "D3Q27|PeriodicHill|mpirun|mpiexec" | grep -v grep)
    if [ -n "$procs" ]; then
        _ph_green "  模擬執行中："
        echo "$procs" | awk '{printf "  PID:%-8s CPU:%5s%% MEM:%5s%% CMD:%s\n",$2,$3,$4,$11}'
    else
        _ph_yellow "  目前沒有偵測到執行中的模擬"
    fi
    echo ""

    # 3) checkrho.dat 最新幾行
    if [ -f "checkrho.dat" ]; then
        _ph_bold "[checkrho.dat] 最新 3 行："
        tail -3 checkrho.dat | sed 's/^/  /'
    fi
    echo ""

    # 4) Ustar_Force_record.dat 最新幾行
    if [ -f "Ustar_Force_record.dat" ]; then
        _ph_bold "[Ustar_Force_record.dat] 最新 3 行："
        tail -3 Ustar_Force_record.dat | sed 's/^/  /'
    fi
    echo ""

    # 5) Result 資料夾統計
    local rdir
    rdir=$(_ph_resultdir 2>/dev/null)
    if [ -n "$rdir" ] && [ -d "$rdir" ]; then
        local vtk_count=$(find "$rdir" -name "*.vtk" -o -name "*.vtr" -o -name "*.vts" -o -name "*.vtu" 2>/dev/null | wc -l)
        local latest_vtk=$(find "$rdir" -name "*.vtk" -o -name "*.vtr" -o -name "*.vts" -o -name "*.vtu" 2>/dev/null | xargs ls -t 2>/dev/null | head -1)
        local dir_size=$(du -sh "$rdir" 2>/dev/null | cut -f1)
        _ph_bold "[Result]"
        echo "  資料夾: $rdir  |  大小: $dir_size  |  VTK 數量: $vtk_count"
        [ -n "$latest_vtk" ] && echo "  最新: $(basename "$latest_vtk")  $(stat -c '%y' "$latest_vtk" 2>/dev/null | cut -d. -f1)"
    fi
    echo ""

    # 6) 磁碟空間
    _ph_bold "[Disk]"
    df -h . | tail -1 | awk '{printf "  使用: %s / %s (%s)  剩餘: %s\n",$3,$2,$5,$4}'

    _ph_cyan "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# ============================================================================
#  Alt+G  →  nvidia-smi（快速看 GPU）
# ============================================================================
_ph_gpu() {
    echo ""
    _ph_cyan "━━━ [Alt+G] GPU 狀態 ━━━"
    nvidia-smi
}

# ============================================================================
#  Alt+C  →  編譯（make clean && make）
# ============================================================================
_ph_compile() {
    echo ""
    _ph_cyan "━━━ [Alt+C] 編譯 ━━━  dir: $(pwd)"
    if [ -f "Makefile" ] || [ -f "makefile" ]; then
        make clean 2>/dev/null
        make -j$(nproc) 2>&1 | tail -20
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            _ph_green "✓ 編譯成功"
        else
            _ph_red "✗ 編譯失敗"
        fi
    else
        _ph_red "✗ 找不到 Makefile"
    fi
}

# ============================================================================
#  Alt+K  →  終止模擬程序
# ============================================================================
_ph_kill() {
    echo ""
    _ph_cyan "━━━ [Alt+K] 終止模擬 ━━━"
    local pids=$(pgrep -f "D3Q27\|PeriodicHill" 2>/dev/null)
    if [ -z "$pids" ]; then
        # 也檢查 mpirun
        pids=$(pgrep -f "mpirun\|mpiexec" 2>/dev/null)
    fi

    if [ -z "$pids" ]; then
        _ph_yellow "找不到執行中的模擬程序"
        return 0
    fi

    _ph_yellow "找到以下程序："
    ps -p $(echo "$pids" | tr '\n' ',') -o pid,etime,cmd --no-headers 2>/dev/null | sed 's/^/  /'
    echo ""
    read -p "  確定要終止嗎？(y/N) " confirm
    if [[ "$confirm" =~ ^[yY]$ ]]; then
        echo "$pids" | xargs kill -15
        sleep 1
        # 檢查是否還活著
        local remaining=$(echo "$pids" | xargs -I{} ps -p {} --no-headers 2>/dev/null)
        if [ -n "$remaining" ]; then
            _ph_yellow "  程序未結束，嘗試 kill -9..."
            echo "$pids" | xargs kill -9
        fi
        _ph_green "✓ 已終止"
    else
        _ph_yellow "取消"
    fi
}

# ============================================================================
#  Alt+L  →  列出 result/ 最新檔案
# ============================================================================
_ph_list_result() {
    echo ""
    _ph_cyan "━━━ [Alt+L] Result 最新檔案 ━━━"
    local rdir
    rdir=$(_ph_resultdir) || return 1
    _ph_bold "$rdir/ 最新 15 個檔案："
    ls -lhtr "$rdir" | tail -15 | sed 's/^/  /'
}

# ============================================================================
#  Alt+D  →  快速 diff checkrho（看密度收斂趨勢）
# ============================================================================
_ph_checkrho_trend() {
    echo ""
    _ph_cyan "━━━ [Alt+D] Density 收斂趨勢 ━━━"
    if [ ! -f "checkrho.dat" ]; then
        _ph_red "✗ 找不到 checkrho.dat"
        return 1
    fi

    local total_lines=$(wc -l < checkrho.dat)
    _ph_bold "checkrho.dat: 共 $total_lines 行"
    echo ""

    # 顯示頭 3 行和尾 10 行
    _ph_yellow "─── 起始 ───"
    head -3 checkrho.dat | sed 's/^/  /'
    _ph_yellow "─── 最新 10 步 ───"
    tail -10 checkrho.dat | sed 's/^/  /'

    # 如果有 awk，算一下平均密度偏差
    echo ""
    _ph_bold "最近 100 步平均密度："
    tail -100 checkrho.dat 2>/dev/null | awk '
    NF>0 {
        for(i=1;i<=NF;i++) {
            if($i+0==$i && $i>0.9 && $i<1.1) { sum+=$i; n++ }
        }
    }
    END { if(n>0) printf "  avg = %.8f  (偏差: %.2e)\n", sum/n, sum/n-1.0; else print "  無法解析" }'
}

# ============================================================================
#  Alt+W  →  一鍵查看 Python 腳本清單 + 執行選單
# ============================================================================
_ph_python_menu() {
    echo ""
    _ph_cyan "━━━ [Alt+W] Python 腳本選單 ━━━  dir: $(pwd)"
    local py_files=($(ls *.py 2>/dev/null | sort))

    if [ ${#py_files[@]} -eq 0 ]; then
        _ph_red "✗ 當前目錄沒有 .py 檔案"
        return 1
    fi

    local i=1
    for f in "${py_files[@]}"; do
        printf "  [%2d] %s\n" $i "$f"
        ((i++))
    done
    echo ""
    read -p "  輸入編號執行（Enter 取消）: " choice
    if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le ${#py_files[@]} ]; then
        local target="${py_files[$((choice-1))]}"
        _ph_yellow "執行: $PH_PYTHON $target"
        $PH_PYTHON "$target"
        _ph_green "✓ $target 完成"
    fi
}

# ============================================================================
#  Alt+B  →  快速備份當前設定檔
# ============================================================================
_ph_backup() {
    echo ""
    _ph_cyan "━━━ [Alt+B] 備份關鍵檔案 ━━━"
    local proj=$(_ph_projname)
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_dir="./backups/${timestamp}"
    mkdir -p "$backup_dir"

    local count=0
    for f in *.h *.cu *.cpp Makefile makefile *.py *.dat *.cfg *.conf *.json *.sh; do
        for g in $f; do
            if [ -f "$g" ]; then
                cp "$g" "$backup_dir/"
                ((count++))
            fi
        done
    done

    _ph_green "✓ 已備份 $count 個檔案到 $backup_dir/"
    du -sh "$backup_dir" | sed 's/^/  /'
}

# ============================================================================
#  Alt+N  →  nohup 啟動模擬（背景執行）
# ============================================================================
_ph_nohup_run() {
    echo ""
    _ph_cyan "━━━ [Alt+N] Nohup 啟動 ━━━"

    # 自動偵測可執行檔
    local exe=""
    for f in ./a.out ./D3Q27* ./periodic_hill* ./main; do
        for g in $f; do
            if [ -x "$g" ]; then
                exe="$g"
                break 2
            fi
        done
    done

    if [ -z "$exe" ]; then
        _ph_red "✗ 找不到可執行檔"
        read -p "  請輸入執行指令: " exe
        [ -z "$exe" ] && return 1
    fi

    _ph_yellow "將執行: nohup mpirun $exe &"
    read -p "  GPU 數量 (預設 4): " ngpu
    ngpu=${ngpu:-4}

    read -p "  確認啟動？(y/N) " confirm
    if [[ "$confirm" =~ ^[yY]$ ]]; then
        nohup mpirun -np "$ngpu" "$exe" > nohup.out 2>&1 &
        _ph_green "✓ 已啟動 (PID: $!)  log → nohup.out"
    else
        _ph_yellow "取消"
    fi
}

# ============================================================================
#  Bind Keys (使用 bash readline bind -x)
# ============================================================================
# 注意：某些 SSH 終端可能攔截 Alt 組合鍵。
# 如果 Alt+X 不生效，可嘗試 ESC+X（按 ESC 再按字母）。

bind -x '"\ef": _ph_run_SL'             # Alt+F → Streamline
bind -x '"\et": _ph_run_MaUTime'        # Alt+T → Ma/U Time
bind -x '"\ep": _ph_run_Benchmark'      # Alt+P → Benchmark
bind -x '"\eh": _ph_download_vtk'       # Alt+H → 下載 VTK
bind -x '"\er": _ph_tail_log'           # Alt+R → 即時監控 log
bind -x '"\es": _ph_status'             # Alt+S → 狀態總覽
bind -x '"\eg": _ph_gpu'                # Alt+G → GPU 狀態
bind -x '"\ec": _ph_compile'            # Alt+C → 編譯
bind -x '"\ek": _ph_kill'               # Alt+K → 終止模擬
bind -x '"\el": _ph_list_result'        # Alt+L → 列出最新結果
bind -x '"\ed": _ph_checkrho_trend'     # Alt+D → Density 趨勢
bind -x '"\ew": _ph_python_menu'        # Alt+W → Python 選單
bind -x '"\eb": _ph_backup'             # Alt+B → 備份
bind -x '"\en": _ph_nohup_run'          # Alt+N → Nohup 啟動

# ============================================================================
#  Help：顯示所有快捷鍵
# ============================================================================
ph_help() {
    echo ""
    _ph_cyan "╔══════════════════════════════════════════════════════════╗"
    _ph_cyan "║        PeriodicHill D3Q27 GILBM 快捷鍵一覽表           ║"
    _ph_cyan "╠══════════════════════════════════════════════════════════╣"
    _ph_cyan "║                                                        ║"
    _ph_cyan "║  ── 執行 Python 腳本 ──                                ║"
    _ph_cyan "║  Alt+F   5.SL_python.py     (Streamline 流線圖)        ║"
    _ph_cyan "║  Alt+T   4.Ma_U_Time.py     (Ma / U 時序監控)          ║"
    _ph_cyan "║  Alt+P   2.Benchmark.py     (ERCOFTAC 基準比對)        ║"
    _ph_cyan "║  Alt+W   Python 腳本選單    (互動式選擇執行)            ║"
    _ph_cyan "║                                                        ║"
    _ph_cyan "║  ── 模擬監控 ──                                        ║"
    _ph_cyan "║  Alt+S   狀態總覽           (GPU/程序/密度/磁碟一覽)    ║"
    _ph_cyan "║  Alt+R   即時監控 log       (tail checkrho/Ustar)      ║"
    _ph_cyan "║  Alt+D   Density 收斂趨勢   (checkrho.dat 分析)        ║"
    _ph_cyan "║  Alt+G   GPU 狀態           (nvidia-smi)               ║"
    _ph_cyan "║                                                        ║"
    _ph_cyan "║  ── 檔案管理 ──                                        ║"
    _ph_cyan "║  Alt+H   下載最新 VTK       (推送/顯示 scp 指令)       ║"
    _ph_cyan "║  Alt+L   列出 result 最新   (最新 15 個檔案)           ║"
    _ph_cyan "║  Alt+B   備份關鍵檔案       (源碼+設定+腳本)           ║"
    _ph_cyan "║                                                        ║"
    _ph_cyan "║  ── 模擬控制 ──                                        ║"
    _ph_cyan "║  Alt+C   編譯               (make clean && make)       ║"
    _ph_cyan "║  Alt+N   Nohup 啟動模擬     (背景 mpirun)              ║"
    _ph_cyan "║  Alt+K   終止模擬           (安全 kill)                ║"
    _ph_cyan "║                                                        ║"
    _ph_cyan "║  提示：Alt 不生效時，可用 ESC+字母 替代                 ║"
    _ph_cyan "║  輸入 ph_help 重新顯示此表                              ║"
    _ph_cyan "╚══════════════════════════════════════════════════════════╝"
    echo ""
}

# ── 載入時顯示提示 ──────────────────────────────────────────────────────────
_ph_green "✓ PeriodicHill 快捷鍵已載入 ($(hostname)) — 輸入 ph_help 查看"
