#!/usr/bin/env bash
# security_layer/install_hook.sh
# Installs the pre-commit secret guard. Run once per clone:
#   bash security_layer/install_hook.sh
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOK="$REPO_ROOT/.git/hooks/pre-commit"

cat > "$HOOK" <<'EOF'
#!/usr/bin/env bash
# Auto-installed by security_layer/install_hook.sh — blocks commits that
# would introduce credentials or forbidden files.
python security_layer/secret_guard.py --staged
EOF

chmod +x "$HOOK"
echo "pre-commit secret guard installed at .git/hooks/pre-commit"
