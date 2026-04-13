#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/run-scripts.sh"

echo "=== Post-Start Hook ==="
run_scripts "$SCRIPT_DIR/post-start.d"
echo "=== Post-Start Complete ==="
