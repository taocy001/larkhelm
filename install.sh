#!/usr/bin/env bash
# larkhelm one-click installer
# Supports: macOS (launchd), Linux (systemd system service / user service)
#
# Usage:
#   ./install.sh                   # auto mode (macOS→launchd; Linux→user service)
#   ./install.sh --local           # install from current directory (dev/test)
#   ./install.sh --mode system     # Linux system service (sudo required)
#   ./install.sh --mode user       # Linux user service (no sudo, default)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATES_DIR="$SCRIPT_DIR/templates"
EXAMPLE_PATH="$SCRIPT_DIR/larkhelm_config.example.json"

# ── Color output ──────────────────────────────────────────────────────────────
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${GREEN}[信息]${NC} $*"; }
warn()  { echo -e "${YELLOW}[警告]${NC} $*"; }
error() { echo -e "${RED}[错误]${NC} $*" >&2; }
step()  { echo -e "\n${CYAN}══ $* ${NC}"; }

# ── Argument parsing ──────────────────────────────────────────────────────────
INSTALL_MODE="user"
LOCAL_INSTALL=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --local)   LOCAL_INSTALL=true; shift ;;
        --mode)
            INSTALL_MODE="${2:-}"
            [[ "$INSTALL_MODE" == "system" || "$INSTALL_MODE" == "user" ]] || {
                error "--mode 只接受 system 或 user"; exit 1; }
            shift 2 ;;
        --help|-h)
            cat <<'HELP'
用法: ./install.sh [选项]

选项:
  --local          从当前目录安装（开发 / 测试用途）
  --mode system    安装为系统服务，开机自启，需要 sudo（仅 Linux）
  --mode user      安装为用户态服务，无需 sudo（Linux 默认）
  --help           显示此帮助

macOS 始终使用 launchd（用户级），无需 --mode。
HELP
            exit 0 ;;
        *) error "未知参数：$1，使用 --help 查看用法"; exit 1 ;;
    esac
done

# ══════════════════════════════════════════════════════════════════════════════
#  Step 0: Augment PATH — cover ARM64 Homebrew and ~/.local/bin before any check
# ══════════════════════════════════════════════════════════════════════════════
for _p in /opt/homebrew/bin /opt/homebrew/sbin /usr/local/bin "$HOME/.local/bin"; do
    [[ -d "$_p" && ":$PATH:" != *":$_p:"* ]] && export PATH="$_p:$PATH"
done

# ══════════════════════════════════════════════════════════════════════════════
#  Step 1: Find Python 3.10+
# ══════════════════════════════════════════════════════════════════════════════
step "检查 Python 版本"

PYTHON3=""
for _py in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$_py" &>/dev/null && \
       "$_py" -c "import sys; assert sys.version_info >= (3,10)" 2>/dev/null; then
        PYTHON3="$_py"
        break
    fi
done

if [[ -z "$PYTHON3" ]]; then
    error "未找到 Python 3.10+，请先安装："
    error "  macOS：  brew install python@3.12"
    error "  Debian： sudo apt-get install python3.12"
    exit 1
fi
info "Python：$($PYTHON3 --version) ✓  ($PYTHON3)"

# ══════════════════════════════════════════════════════════════════════════════
#  Step 2: Ensure pipx is available
# ══════════════════════════════════════════════════════════════════════════════
step "检查 pipx"

_install_pipx() {
    if command -v apt-get &>/dev/null; then
        info "通过 apt 安装 pipx..."
        sudo apt-get install -y pipx
    elif command -v brew &>/dev/null; then
        info "通过 Homebrew 安装 pipx..."
        brew install pipx
        # Homebrew may install to a prefix not yet in PATH — refresh
        BREW_PREFIX=$(brew --prefix 2>/dev/null || echo "/usr/local")
        [[ ":$PATH:" != *":$BREW_PREFIX/bin:"* ]] && export PATH="$BREW_PREFIX/bin:$PATH"
    else
        info "通过 pip --user 安装 pipx..."
        "$PYTHON3" -m pip install --user pipx
    fi
    # Ensure the pipx bin dir is in PATH for the rest of this script
    "$PYTHON3" -m pipx ensurepath 2>/dev/null || true
    export PATH="$HOME/.local/bin:$PATH"
}

if ! command -v pipx &>/dev/null; then
    warn "未找到 pipx，正在安装..."
    _install_pipx
fi

# Final check (covers both fresh-install and already-installed paths)
if ! command -v pipx &>/dev/null; then
    error "pipx 安装失败，请手动安装后重试：https://pipx.pypa.io/"
    exit 1
fi
info "pipx：$(pipx --version) ✓"

# ══════════════════════════════════════════════════════════════════════════════
#  Step 3: Install / upgrade larkhelm
# ══════════════════════════════════════════════════════════════════════════════
step "安装 larkhelm"

if [[ "$LOCAL_INSTALL" == "true" ]]; then
    info "从本地目录安装（开发模式）..."
    if pipx list 2>/dev/null | grep -q "larkhelm"; then
        pipx uninstall larkhelm
    fi
    pipx install "$SCRIPT_DIR"
else
    info "从 PyPI 安装..."
    if pipx list 2>/dev/null | grep -q "larkhelm"; then
        info "检测到已安装版本，执行升级..."
        pipx upgrade larkhelm
    else
        pipx install larkhelm
    fi
fi

# Resolve the actual binary path
PIPX_BIN_DIR="${PIPX_BIN_DIR:-}"
if [[ -z "$PIPX_BIN_DIR" ]]; then
    PIPX_BIN_DIR="$(pipx environment 2>/dev/null \
        | grep '^PIPX_BIN_DIR=' | grep -v '=$' | cut -d= -f2 | tail -n1)"
fi
PIPX_BIN_DIR="${PIPX_BIN_DIR:-$HOME/.local/bin}"
LARKHELM_BIN="$PIPX_BIN_DIR/larkhelm"

[[ -x "$LARKHELM_BIN" ]] || {
    error "安装后未找到可执行文件：$LARKHELM_BIN"
    exit 1
}
info "larkhelm 安装完成 ✓  ($LARKHELM_BIN)"

# ══════════════════════════════════════════════════════════════════════════════
#  Step 4: Config file setup
# ══════════════════════════════════════════════════════════════════════════════
step "配置文件"

XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$HOME/.config}"
CONFIG_DIR="$XDG_CONFIG_HOME/larkhelm"
CONFIG_PATH="$CONFIG_DIR/config.json"

XDG_DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
DATA_DIR="$XDG_DATA_HOME/larkhelm"
LOG_DIR="$DATA_DIR/logs"

mkdir -p "$CONFIG_DIR" "$LOG_DIR"

if [[ ! -f "$CONFIG_PATH" ]]; then
    [[ -f "$EXAMPLE_PATH" ]] || { error "未找到配置模板：$EXAMPLE_PATH"; exit 1; }
    cp "$EXAMPLE_PATH" "$CONFIG_PATH"
    chmod 600 "$CONFIG_PATH"
    warn "已从模板创建配置文件：$CONFIG_PATH"
    warn "请填写 APP_ID 和 APP_SECRET 后重新运行此脚本。"
    exit 0
fi

# Ensure only owner can read config (larkhelm refuses world-readable config)
chmod 600 "$CONFIG_PATH"

# Validate JSON
"$PYTHON3" -c "import json,sys; json.load(open('$CONFIG_PATH'))" 2>/dev/null || {
    error "配置文件 JSON 格式不合法：$CONFIG_PATH"
    exit 1
}

# Check required fields
APP_ID_VAL=$("$PYTHON3" -c "import json; print(json.load(open('$CONFIG_PATH')).get('APP_ID',''))" 2>/dev/null || echo "")
APP_SECRET_VAL=$("$PYTHON3" -c "import json; print(json.load(open('$CONFIG_PATH')).get('APP_SECRET',''))" 2>/dev/null || echo "")

if [[ "$APP_ID_VAL" == "YOUR_APP_ID"   || -z "$APP_ID_VAL" || \
      "$APP_SECRET_VAL" == "YOUR_APP_SECRET" || -z "$APP_SECRET_VAL" ]]; then
    error "请先在 $CONFIG_PATH 中填写真实的 APP_ID 和 APP_SECRET，然后重新运行 install.sh"
    exit 1
fi
info "配置校验通过 ✓"

# ══════════════════════════════════════════════════════════════════════════════
#  Step 5: AI CLI availability check
# ══════════════════════════════════════════════════════════════════════════════
step "检查 AI CLI"

_cli_found=0
for _cli in claude gemini kimi deepseek; do
    if command -v "$_cli" &>/dev/null; then
        info "$_cli ✓  ($(command -v "$_cli"))"
        (( _cli_found++ )) || true
    else
        warn "$_cli 未找到 — 若要使用该 backend，请安装后加入 PATH 或在配置中设置 ${_cli}_command"
    fi
done

if [[ "$_cli_found" -eq 0 ]]; then
    # deepseek runs via HTTP API, no CLI needed — check if it's configured
    _ds_key=$("$PYTHON3" -c "import json; d=json.load(open('$CONFIG_PATH')); print(d.get('DEEPSEEK_API_KEY',''))" 2>/dev/null || echo "")
    if [[ -z "$_ds_key" ]]; then
        error "未找到任何 AI CLI（claude/gemini/kimi）且未配置 DEEPSEEK_API_KEY。"
        error "至少需要一个 backend 才能运行 larkhelm。"
        error "推荐安装 Claude CLI：https://docs.anthropic.com/claude/docs/claude-cli"
        exit 1
    fi
    info "deepseek HTTP API 已配置 ✓"
fi

# ══════════════════════════════════════════════════════════════════════════════
#  Helper: substitute placeholders in a service/plist template
# ══════════════════════════════════════════════════════════════════════════════
_apply_template() {
    local src="$1" dst="$2"
    local user_val="${3:-}"   # empty = user service (delete User/Home/Path lines)
    local wantedby="${4:-default.target}"
    local workdir="${5:-$HOME}"
    local home_val="${6:-$HOME}"
    local path_val="${7:-$PATH}"

    cp "$src" "$dst"

    # Common substitutions
    sed -i.bak \
        -e "s|__LARKHELM_BIN__|$LARKHELM_BIN|g" \
        -e "s|__CONFIG__|$CONFIG_PATH|g"          \
        -e "s|__DATA_DIR__|$DATA_DIR|g"           \
        -e "s|__LOG_DIR__|$LOG_DIR|g"             \
        -e "s|__WANTEDBY__|$wantedby|g"           \
        -e "s|__WORKDIR__|$workdir|g"             \
        -e "s|__HOME__|$home_val|g"               \
        -e "s|__PATH__|$path_val|g"               \
        "$dst"
    rm -f "$dst.bak"

    if [[ -n "$user_val" ]]; then
        # System service: fill in User, Home, Path lines
        sed -i.bak \
            -e "s|^__USER_LINE__$|User=$user_val|"           \
            -e "s|^__HOME_LINE__$|Environment=\"HOME=$home_val\"|" \
            -e "s|^__PATH_LINE__$|Environment=\"PATH=$path_val\"|" \
            "$dst"
    else
        # User service: drop those lines (user services inherit the environment)
        sed -i.bak \
            -e '/^__USER_LINE__$/d' \
            -e '/^__HOME_LINE__$/d' \
            -e '/^__PATH_LINE__$/d' \
            "$dst"
    fi
    rm -f "$dst.bak"
}

# ══════════════════════════════════════════════════════════════════════════════
#  Helper: configure logrotate (Linux only)
# ══════════════════════════════════════════════════════════════════════════════
_setup_logrotate() {
    local logrotate_src="$TEMPLATES_DIR/larkhelm.logrotate"
    local logrotate_dst="/etc/logrotate.d/larkhelm"

    [[ -f "$logrotate_src" ]] || { warn "未找到 logrotate 模板，跳过"; return; }

    local tmp_lr
    tmp_lr=$(mktemp)
    sed "s|__LOG_DIR__|$LOG_DIR|g" "$logrotate_src" > "$tmp_lr"

    if [[ "$(id -u)" -eq 0 ]] || [[ -w "/etc/logrotate.d" ]]; then
        cp "$tmp_lr" "$logrotate_dst"
        chmod 644 "$logrotate_dst"
        rm -f "$tmp_lr"
        info "logrotate 已配置 ✓  ($logrotate_dst)"
    elif command -v sudo &>/dev/null && sudo cp "$tmp_lr" "$logrotate_dst" 2>/dev/null \
         && sudo chmod 644 "$logrotate_dst"; then
        rm -f "$tmp_lr"
        info "logrotate 已配置（via sudo）✓"
    else
        info "提示：运行以下命令启用日志轮转（需要 sudo）："
        info "  sudo cp $tmp_lr $logrotate_dst"
        # leave tmp_lr for user to copy manually
    fi
}

# ══════════════════════════════════════════════════════════════════════════════
#  Step 6: Register service
# ══════════════════════════════════════════════════════════════════════════════
step "注册服务"

OS_TYPE=$(uname -s)
CURRENT_USER="${SUDO_USER:-$(whoami)}"

if [[ "$OS_TYPE" == "Darwin" ]]; then
    # ── macOS: launchd (user agent, always) ──────────────────────────────────
    info "macOS 检测到，安装 launchd agent..."

    PLIST_SRC="$TEMPLATES_DIR/com.larkhelm.plist"
    PLIST_DST="$HOME/Library/LaunchAgents/com.larkhelm.plist"

    [[ -f "$PLIST_SRC" ]] || { error "未找到 plist 模板：$PLIST_SRC"; exit 1; }

    mkdir -p "$HOME/Library/LaunchAgents"

    # Build PATH for launchd: include ARM64 Homebrew + common tool locations
    LAUNCHD_PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PIPX_BIN_DIR"
    # Also include user's current PATH entries that aren't already covered
    IFS=: read -ra _extra_paths <<< "$PATH"
    for _ep in "${_extra_paths[@]}"; do
        [[ ":$LAUNCHD_PATH:" != *":$_ep:"* ]] && LAUNCHD_PATH="$LAUNCHD_PATH:$_ep"
    done

    cp "$PLIST_SRC" "$PLIST_DST"
    # macOS sed requires '' after -i (no backup extension)
    sed -i '' \
        -e "s|__LARKHELM_BIN__|$LARKHELM_BIN|g" \
        -e "s|__CONFIG__|$CONFIG_PATH|g"          \
        -e "s|__DATA_DIR__|$DATA_DIR|g"           \
        -e "s|__LOG_DIR__|$LOG_DIR|g"             \
        -e "s|__HOME__|$HOME|g"                   \
        -e "s|__PATH__|$LAUNCHD_PATH|g"           \
        "$PLIST_DST"

    # Reload agent (unload first if already registered)
    launchctl bootout "gui/$(id -u)" "$PLIST_DST" 2>/dev/null || true
    launchctl bootstrap "gui/$(id -u)" "$PLIST_DST"

    info "launchd agent 已安装并启动 ✓"
    info ""
    info "  状态：launchctl list | grep larkhelm"
    info "  日志：tail -f $LOG_DIR/larkhelm.log"
    info ""
    info "服务在您登录时自动运行，注销后停止。"
    info "（若需要开机自启无需登录，需要使用系统级 launchd daemon — 联系管理员配置。）"

elif [[ "$OS_TYPE" == "Linux" ]]; then

    SERVICE_SRC="$TEMPLATES_DIR/larkhelm.service"
    [[ -f "$SERVICE_SRC" ]] || { error "未找到服务模板：$SERVICE_SRC"; exit 1; }

    if [[ "$INSTALL_MODE" == "system" ]]; then
        # ── Linux: system service ─────────────────────────────────────────────
        info "安装为系统服务 (/etc/systemd/system/)..."

        if [[ "$(id -u)" -ne 0 ]]; then
            error "安装系统服务需要 root 权限，请使用 sudo 运行："
            error "  sudo $0 --mode system"
            error "或改用用户态服务（无需 sudo）："
            error "  $0 --mode user"
            exit 1
        fi

        CURRENT_HOME=$(eval echo "~$CURRENT_USER")

        # Stop and remove any user service to avoid running two instances
        USER_SERVICE_FILE="$CURRENT_HOME/.config/systemd/user/larkhelm.service"
        if [[ -f "$USER_SERVICE_FILE" ]]; then
            info "检测到用户态服务，正在清理..."
            systemctl --user -M "${CURRENT_USER}@.host" stop    larkhelm 2>/dev/null || true
            systemctl --user -M "${CURRENT_USER}@.host" disable larkhelm 2>/dev/null || true
            rm -f "$USER_SERVICE_FILE"
            systemctl --user -M "${CURRENT_USER}@.host" daemon-reload 2>/dev/null || true
        fi

        # Determine the user's PATH for the service environment
        USER_PATH=$(su -l "$CURRENT_USER" -s /bin/sh -c 'echo $PATH' 2>/dev/null || echo "$PATH")
        # Ensure pipx bin dir is included
        [[ ":$USER_PATH:" != *":$PIPX_BIN_DIR:"* ]] && USER_PATH="$PIPX_BIN_DIR:$USER_PATH"

        SERVICE_DST="/etc/systemd/system/larkhelm.service"
        _apply_template "$SERVICE_SRC" "$SERVICE_DST" \
            "$CURRENT_USER" "multi-user.target" "$DATA_DIR" "$CURRENT_HOME" "$USER_PATH"

        systemctl daemon-reload
        systemctl enable  larkhelm
        systemctl restart larkhelm

        _setup_logrotate

        info "系统服务已安装并启动 ✓（开机自启，无需 linger）"
        info ""
        info "  状态：sudo systemctl status larkhelm"
        info "  停止：sudo systemctl stop larkhelm"
        info "  日志：tail -f $LOG_DIR/larkhelm.log"

    else
        # ── Linux: user service ───────────────────────────────────────────────
        info "安装为用户态服务 (~/.config/systemd/user/)..."

        # ── XDG_RUNTIME_DIR check (needed for user systemd) ──────────────────
        if [[ -z "${XDG_RUNTIME_DIR:-}" ]]; then
            _xrd="/run/user/$(id -u)"
            if [[ -d "$_xrd" ]]; then
                export XDG_RUNTIME_DIR="$_xrd"
                info "已自动设置 XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR"
            else
                error "user systemd 不可用（XDG_RUNTIME_DIR 未设置）。"
                error ""
                error "这通常发生在 SSH 无图形界面会话中。解决方案（任选一）："
                error ""
                error "  1. 设置环境变量后重试："
                error "     export XDG_RUNTIME_DIR=/run/user/\$(id -u)"
                error "     ./install.sh"
                error ""
                error "  2. 改用系统服务（需要 sudo，开机自启）："
                error "     sudo ./install.sh --mode system"
                exit 1
            fi
        fi

        # Smoke-test that user systemd bus is reachable
        if ! systemctl --user status &>/dev/null 2>&1; then
            # Try to recover by exporting the socket path explicitly
            _bus="/run/user/$(id -u)/bus"
            if [[ -S "$_bus" ]]; then
                export DBUS_SESSION_BUS_ADDRESS="unix:path=$_bus"
                info "已自动设置 DBUS_SESSION_BUS_ADDRESS"
            fi
            # Retest
            if ! systemctl --user status &>/dev/null 2>&1; then
                error "user systemd 不可访问。"
                error "建议：sudo ./install.sh --mode system"
                exit 1
            fi
        fi

        # Clean up any conflicting system service
        if [[ -f "/etc/systemd/system/larkhelm.service" ]]; then
            info "检测到系统服务，正在停止（需要 sudo）..."
            if sudo systemctl stop    larkhelm 2>/dev/null && \
               sudo systemctl disable larkhelm 2>/dev/null && \
               sudo rm -f "/etc/systemd/system/larkhelm.service" && \
               sudo systemctl daemon-reload; then
                info "系统服务已清理 ✓"
            else
                warn "无法自动清理系统服务，请手动运行："
                warn "  sudo systemctl disable --now larkhelm"
                warn "  sudo rm /etc/systemd/system/larkhelm.service"
            fi
        fi

        SYSTEMD_USER_DIR="$HOME/.config/systemd/user"
        SERVICE_DST="$SYSTEMD_USER_DIR/larkhelm.service"
        mkdir -p "$SYSTEMD_USER_DIR"

        _apply_template "$SERVICE_SRC" "$SERVICE_DST" \
            "" "default.target" "$HOME" "$HOME" "$PATH"

        systemctl --user daemon-reload
        systemctl --user enable  larkhelm
        systemctl --user restart larkhelm

        # ── Enable linger so the service survives logout ──────────────────────
        if ! loginctl show-user "$CURRENT_USER" 2>/dev/null | grep -q 'Linger=yes'; then
            info "尝试启用 systemd linger（注销后服务保持运行）..."
            if loginctl enable-linger "$CURRENT_USER" 2>/dev/null; then
                info "linger 已启用 ✓"
            elif sudo loginctl enable-linger "$CURRENT_USER" 2>/dev/null; then
                info "linger 已启用（via sudo）✓"
            else
                warn "无法自动启用 linger，服务将在注销后停止。"
                warn "手动启用（任选一）："
                warn "  loginctl enable-linger $CURRENT_USER"
                warn "  sudo loginctl enable-linger $CURRENT_USER"
                warn "或改用系统服务：sudo ./install.sh --mode system"
            fi
        else
            info "linger 已启用 ✓"
        fi

        _setup_logrotate

        info "用户态服务已安装并启动 ✓"
        info ""
        info "  状态：systemctl --user status larkhelm"
        info "  停止：systemctl --user stop larkhelm"
        info "  日志：tail -f $LOG_DIR/larkhelm.log"
    fi

else
    error "不支持的操作系统：$OS_TYPE（支持 macOS / Linux）"
    exit 1
fi

echo ""
info "安装完成 🎉  飞书 AI 桥接服务已在后台运行。"
