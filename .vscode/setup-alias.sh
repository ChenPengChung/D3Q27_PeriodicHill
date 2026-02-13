#!/usr/bin/env bash
#
# Setup 'mobaxterm' command alias for Mac/Linux
# Run: source ./.vscode/setup-alias.sh
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFDLAB_SCRIPT="$SCRIPT_DIR/cfdlab-mac.sh"

# Determine shell profile
if [[ -n "$ZSH_VERSION" ]]; then
    SHELL_PROFILE="$HOME/.zshrc"
    SHELL_NAME="zsh"
elif [[ -n "$BASH_VERSION" ]]; then
    if [[ -f "$HOME/.bash_profile" ]]; then
        SHELL_PROFILE="$HOME/.bash_profile"
    else
        SHELL_PROFILE="$HOME/.bashrc"
    fi
    SHELL_NAME="bash"
else
    SHELL_PROFILE="$HOME/.profile"
    SHELL_NAME="sh"
fi

# Alias code to add
ALIAS_CODE="
# ========== MobaXterm Alias ==========
alias mobaxterm='$CFDLAB_SCRIPT'
# ========== End MobaXterm Alias =========="

# Check if already added
if grep -q "MobaXterm Alias" "$SHELL_PROFILE" 2>/dev/null; then
    echo "[INFO] 'mobaxterm' alias already exists in $SHELL_PROFILE"
else
    echo "$ALIAS_CODE" >> "$SHELL_PROFILE"
    echo "[SUCCESS] 'mobaxterm' alias added to $SHELL_PROFILE"
fi

# Activate immediately
alias mobaxterm="$CFDLAB_SCRIPT"
echo ""
echo "[READY] You can now use:"
echo "  mobaxterm gpus"
echo "  mobaxterm ssh 87:3"
echo "  mobaxterm status"
echo ""
