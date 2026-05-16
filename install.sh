#!/usr/bin/env bash
# larkhelm one-click installer (pipx bootstrap method)
# Supports: macOS (launchd), Linux (systemd system service / user service)
#
# Usage:
#   ./install.sh                   # Install from PyPI; Linux default: user service
#   ./install.sh --local           # Install from current directory (development mode)
#   ./install.sh --mode system     # Install as system service (requires sudo, Linux only)
#   ./install.sh --mode user       # Install as user service (no sudo required)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATES_DIR="$SCRIPT_DIR/templates"
EXAMPLE_PATH="$SCRIPT_DIR/larkhelm_config.example.json"

# ── Color output ──────────────────────────────────────────
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'; NC='\033[0m'
info()  { echo -e "${GREEN}[信息]${NC} $*"; }
warn()  { echo -e "${YELLOW}[警告]${NC} $*"; }
error() { echo -e "${RED}[错误]${NC} $*" >&2; }

# ── Argument parsing ──────────────────────────────────────────
INSTALL_MODE="user"   # Default: user service
LOCAL_INSTALL=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --local)
            LOCAL_INSTALL=true
            shift
            ;;
        --mode)
            INSTALL_MODE="${2:-}"
            if [[ "$INSTALL_MODE" != "system" && "$INSTALL_MODE" != "user" ]]; then
                error "--mode 只接受 system 或 user"
                exit 1
            fi
            shift 2
            ;;
        --help|-h)
            echo "用法: $0 [--local] [--mode system|user]"
            echo "  --local   从当前目录安装（开发/测试用途）"
            echo "  system    安装为系统服务，开机自启，需要 sudo（仅 Linux）"
            echo "  user      安装为用户态服务，无需 sudo（默认）"
            exit 0
            ;;
        *)
            error "未知参数：$1，使用 --help 查看用法"
            exit 1
            ;;
    esac
done

# ══════════════════════════════════════════════════════
#  Step 0: Python version check (requires 3.10+)
# ══════════════════════════════════════════════════════
if ! python3 -c "import sys; assert sys.version_info >= (3,10)" 2>/dev/null; then
    PYVER=$(python3 --version 2>&1 || echo "未找到")
    error "Python 版本不足：$PYVER（需要 3.10+）"
    error "  macOS：brew install python@3.12"
    error "  Debian：sudo apt-get install python3.12"
    exit 1
fi
info "Python 版本：$(python3 --version) ✓"

# ══════════════════════════════════════════════════════
#  Step 1: Ensure pipx is available
# ══════════════════════════════════════════════════════
if ! command -v pipx &>/dev/null; then
    warn "未找到 pipx，正在安装..."
    if command -v apt-get &>/dev/null; then
        sudo apt-get install -y pipx
    elif command -v brew &>/dev/null; then
        brew install pipx
    else
        python3 -m pip install --user pipx
    fi
    # Ensure pipx bin directory is in PATH
    python3 -m pipx ensurepath
    export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v pipx &>/dev/null; then
    error "pipx 安装失败，请手动安装后重试：https://pipx.pypa.io/"
    exit 1
fi
info "pipx 版本：$(pipx --version) ✓"

# ══════════════════════════════════════════════════════
#  Step 2: Install larkhelm
# ══════════════════════════════════════════════════════
if [[ "$LOCAL_INSTALL" == "true" ]]; then
    info "从本地目录正式安装..."
    if (cd /tmp && pipx list | grep -q "larkhelm"); then
        (cd /tmp && pipx uninstall larkhelm)
    fi
    pipx install "$SCRIPT_DIR"
else
    info "从 PyPI 安装 larkhelm..."
    if (cd /tmp && pipx list | grep -q "larkhelm"); then
        info "已检测到已安装版本，执行升级..."
        (cd /tmp && pipx upgrade larkhelm)
    else
        pipx install larkhelm
    fi
fi
info "larkhelm 安装/升级完成 ✓"

# Determine pipx bin directory (prefer environment variable for compatibility across pipx versions)
PIPX_BIN_DIR="${PIPX_BIN_DIR:-}"
if [[ -z "$PIPX_BIN_DIR" ]]; then
    # pipx >= 1.0: pipx environment outputs KEY=VALUE format; filter out blank lines and bare KEY= entries
    PIPX_BIN_DIR="$(pipx environment 2>/dev/null | grep '^PIPX_BIN_DIR=' | grep -v '=$' | cut -d= -f2 | tail -n 1)"
fi
PIPX_BIN_DIR="${PIPX_BIN_DIR:-$HOME/.local/bin}"
LARKHELM_BIN="$PIPX_BIN_DIR/larkhelm"
if [[ ! -x "$LARKHELM_BIN" ]]; then
    error "安装后未找到可执行文件：$LARKHELM_BIN"
    exit 1
fi

# ══════════════════════════════════════════════════════
#  Step 3: Guide user to create the config file
# ══════════════════════════════════════════════════════
XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$HOME/.config}"
CONFIG_DIR="$XDG_CONFIG_HOME/larkhelm"
CONFIG_PATH="$CONFIG_DIR/config.json"

XDG_DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
DATA_DIR="$XDG_DATA_HOME/larkhelm"
LOG_DIR="$DATA_DIR/logs"

mkdir -p "$CONFIG_DIR"
mkdir -p "$LOG_DIR"

if [[ ! -f "$CONFIG_PATH" ]]; then
    if [[ -f "$EXAMPLE_PATH" ]]; then
        cp "$EXAMPLE_PATH" "$CONFIG_PATH"
        warn "已从模板创建配置文件：$CONFIG_PATH"
        warn "请填写 APP_ID 和 APP_SECRET 后重新运行此脚本或手动启动服务。"
    else
        error "未找到配置模板：$EXAMPLE_PATH"
        exit 1
    fi
fi

# Validate config file contents
if ! python3 -c "import json,sys; json.load(open('$CONFIG_PATH'))" 2>/dev/null; then
    error "配置文件 JSON 格式不合法：$CONFIG_PATH"
    exit 1
fi

APP_ID_VAL=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH')).get('APP_ID',''))")
APP_SECRET_VAL=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH')).get('APP_SECRET',''))")

if [[ "$APP_ID_VAL" == "YOUR_APP_ID" || -z "$APP_ID_VAL" || \
      "$APP_SECRET_VAL" == "YOUR_APP_SECRET" || -z "$APP_SECRET_VAL" ]]; then
    error "请先在 $CONFIG_PATH 中填写真实的 APP_ID 和 APP_SECRET，然后重新运行 install.sh"
    exit 1
fi
info "配置校验通过 ✓"

# ══════════════════════════════════════════════════════
#  CLI tool check (warn only, do not abort)
# ══════════════════════════════════════════════════════
if ! command -v claude &>/dev/null; then
    warn "未找到 claude CLI，请安装后加入 PATH，或在配置文件中设置 claude_command。"
fi
if ! command -v gemini &>/dev/null; then
    warn "未找到 gemini CLI，请安装后加入 PATH，或在配置文件中设置 gemini_command。"
fi

# ══════════════════════════════════════════════════════
#  Step 4: Register system service
# ══════════════════════════════════════════════════════
OS_TYPE=$(uname -s)
CURRENT_USER="${SUDO_USER:-$(whoami)}"

if [[ "$OS_TYPE" == "Darwin" ]]; then
    # ── macOS: launchd (always user mode) ────────────────────
    info "检测到 macOS，安装 launchd agent..."

    PLIST_SRC="$TEMPLATES_DIR/com.larkhelm.plist"
    PLIST_DST="$HOME/Library/LaunchAgents/com.larkhelm.plist"

    [[ -f "$PLIST_SRC" ]] || { error "未找到服务模板：$PLIST_SRC"; exit 1; }

    mkdir -p "$HOME/Library/LaunchAgents"
    cp "$PLIST_SRC" "$PLIST_DST"
    sed -i '' "s|PIPX_BIN_PLACEHOLDER|$PIPX_BIN_DIR|g" "$PLIST_DST"
    sed -i '' "s|CONFIG_PLACEHOLDER|$CONFIG_PATH|g"     "$PLIST_DST"
    sed -i '' "s|DATA_DIR_PLACEHOLDER|$DATA_DIR|g"      "$PLIST_DST"
    sed -i '' "s|LOG_DIR_PLACEHOLDER|$LOG_DIR|g"        "$PLIST_DST"
    sed -i '' "s|PATH_PLACEHOLDER|$PATH|g"              "$PLIST_DST"

    launchctl bootout "gui/$(id -u)" "$PLIST_DST" 2>/dev/null || true
    launchctl bootstrap "gui/$(id -u)" "$PLIST_DST"

    info "launchd agent 已安装并启动 ✓"
    echo ""
    info "查看状态：launchctl list | grep larkhelm"
    info "查看日志：tail -f $LOG_DIR/larkhelm.log"

elif [[ "$OS_TYPE" == "Linux" ]]; then

    SERVICE_SRC="$TEMPLATES_DIR/larkhelm.service"
    [[ -f "$SERVICE_SRC" ]] || { error "未找到服务模板：$SERVICE_SRC"; exit 1; }

    if [[ "$INSTALL_MODE" == "system" ]]; then
        # ── Linux: system service (--mode system) ────────────
        info "安装为系统服务（/etc/systemd/system/）..."

        if [[ "$(id -u)" -ne 0 ]]; then
            error "安装系统服务需要 root 权限，请使用 sudo 运行："
            error "  sudo $0 --mode system"
            error "或改用用户态服务（无需 sudo）："
            error "  $0 --mode user"
            exit 1
        fi

        # Clean up any existing user service (avoid running two instances)
        CURRENT_HOME=$(eval echo "~$CURRENT_USER")
        USER_SERVICE_FILE="$CURRENT_HOME/.config/systemd/user/larkhelm.service"
        if [[ -f "$USER_SERVICE_FILE" ]]; then
            info "检测到用户态服务，正在停止并清理..."
            systemctl --user --machine="${CURRENT_USER}@.host" stop    larkhelm 2>/dev/null || true
            systemctl --user --machine="${CURRENT_USER}@.host" disable larkhelm 2>/dev/null || true
            rm -f "$USER_SERVICE_FILE"
            systemctl --user --machine="${CURRENT_USER}@.host" daemon-reload 2>/dev/null || true
            info "用户态服务已清理 ✓"
        fi

        SERVICE_DST="/etc/systemd/system/larkhelm.service"
        cp "$SERVICE_SRC" "$SERVICE_DST"
        sed -i "s|PIPX_BIN_PLACEHOLDER|$PIPX_BIN_DIR|g" "$SERVICE_DST"
        sed -i "s|CONFIG_PLACEHOLDER|$CONFIG_PATH|g"     "$SERVICE_DST"
        sed -i "s|DATA_DIR_PLACEHOLDER|$DATA_DIR|g"      "$SERVICE_DST"
        sed -i "s|LOG_DIR_PLACEHOLDER|$LOG_DIR|g"        "$SERVICE_DST"

        # System service requires specifying the running user and their PATH
        # Note: GNU sed /a does not support \n for multi-line append; split into two calls
        USER_PATH=$(su -l "$CURRENT_USER" -s /bin/sh -c 'echo $PATH' 2>/dev/null || echo "$PATH")
        sed -i "/^Type=simple/a User=$CURRENT_USER" "$SERVICE_DST"
        sed -i "/^User=$CURRENT_USER/a Environment=\"PATH=$USER_PATH\"" "$SERVICE_DST"

        systemctl daemon-reload
        systemctl enable larkhelm
        systemctl restart larkhelm

        info "系统服务已安装并启动 ✓（开机自启，无需 linger）"
        echo ""
        info "查看状态：sudo systemctl status larkhelm"
        info "停止服务：sudo systemctl stop larkhelm"
        info "查看日志：tail -f $LOG_DIR/larkhelm.log"

    else
        # ── Linux: user service (default) ──────────────────
        info "安装为用户态服务（~/.config/systemd/user/）..."

        # Clean up any existing system service (avoid running two instances)
        if systemctl is-active larkhelm &>/dev/null || \
           [[ -f "/etc/systemd/system/larkhelm.service" ]]; then
            info "检测到系统服务，正在停止并清理（需要 sudo）..."
            if [[ "$(id -u)" -eq 0 ]]; then
                systemctl stop    larkhelm 2>/dev/null || true
                systemctl disable larkhelm 2>/dev/null || true
                rm -f "/etc/systemd/system/larkhelm.service"
                systemctl daemon-reload
            else
                warn "检测到系统服务但无 sudo 权限，无法自动清理。"
                warn "请手动运行：sudo systemctl disable --now larkhelm"
            fi
        fi

        SYSTEMD_USER_DIR="$HOME/.config/systemd/user"
        SERVICE_DST="$SYSTEMD_USER_DIR/larkhelm.service"

        mkdir -p "$SYSTEMD_USER_DIR"
        cp "$SERVICE_SRC" "$SERVICE_DST"
        sed -i "s|PIPX_BIN_PLACEHOLDER|$PIPX_BIN_DIR|g" "$SERVICE_DST"
        sed -i "s|CONFIG_PLACEHOLDER|$CONFIG_PATH|g"     "$SERVICE_DST"
        sed -i "s|DATA_DIR_PLACEHOLDER|$DATA_DIR|g"      "$SERVICE_DST"
        sed -i "s|LOG_DIR_PLACEHOLDER|$LOG_DIR|g"        "$SERVICE_DST"

        systemctl --user daemon-reload
        systemctl --user enable larkhelm
        systemctl --user restart larkhelm

        info "用户态服务已启用并启动 ✓"

        # linger check
        if ! loginctl show-user "$CURRENT_USER" 2>/dev/null | grep -q 'Linger=yes'; then
            cat >&2 <<'LINGEREOF'

┌─────────────────────────────────────────────────────┐
│  警告：systemd linger 未启用                        │
│  注销或重启后服务将被终止。                         │
│                                                     │
│  修复方法（需要 sudo）：                            │
│    loginctl enable-linger $(whoami)                 │
│                                                     │
│  或改用系统服务（开机自启，无此问题）：             │
│    sudo ./install.sh --mode system                  │
└─────────────────────────────────────────────────────┘

LINGEREOF
        fi

        echo ""
        info "查看状态：systemctl --user status larkhelm"
        info "查看日志：tail -f $LOG_DIR/larkhelm.log"
    fi

else
    error "不支持的操作系统：$OS_TYPE（支持 macOS / Linux）"
    exit 1
fi

echo ""
# Single source of truth for the logrotate hint — emitted after both
# system / user branches (and macOS, which is a no-op since logrotate
# is a Linux package). Was duplicated in both Linux branches; consolidated
# here so the operator sees it exactly once regardless of install mode.
if [[ "$OS_TYPE" == "Linux" ]]; then
    info "提示：运行 sudo cp templates/larkhelm.logrotate /etc/logrotate.d/larkhelm 并把 LOG_DIR_PLACEHOLDER 替换为 $LOG_DIR 以启用日志轮转"
fi
info "安装完成！飞书 AI 桥接服务已在后台运行。"
