#!/bin/bash
# Sanity check AppWorld task data before launching a batch.
# Catches the leftover-0-byte-db corruption pattern that caused 9dabbc9_2 to
# permanently fail since 2026-05-09 (root cause: aborted batch wrote 0-byte
# venmo.db that overrode base_dbs/venmo.db, causing "no such table: users"
# every load).
#
# Usage: bash scripts/check_appworld_data_health.sh
# Exit 0 = clean, exit 1 = found leftover (will list).

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
DEFAULT_APPWORLD_ROOT="$REPO_ROOT/../appworld"
APPWORLD_ROOT=${APPWORLD_ROOT:-$DEFAULT_APPWORLD_ROOT}
APPWORLD_DATA="$APPWORLD_ROOT/data/tasks"

if [ ! -d "$APPWORLD_DATA" ]; then
    echo "ERROR: $APPWORLD_DATA does not exist"
    exit 2
fi

leftovers=$(find "$APPWORLD_DATA" -name "*.db" -size 0 2>/dev/null)
if [ -n "$leftovers" ]; then
    echo "⚠️  Found 0-byte .db leftover files (corrupt — override base_dbs):"
    echo "$leftovers"
    echo
    echo "Fix: delete these files, then RESTART the RPC server to clear in-memory cache:"
    echo "  rm <files above>"
    echo "  kill \$(lsof -ti :4242) && cd \"$APPWORLD_ROOT\" && .venv/bin/python rpc/server.py"
    exit 1
fi

echo "✓ AppWorld task data clean (no 0-byte .db leftovers in $APPWORLD_DATA)"
exit 0
