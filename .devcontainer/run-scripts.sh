#!/bin/bash
set -euo pipefail

# Shared runner library for DevContainer lifecycle hooks.
# Sources all *.sh scripts (excluding *.skip.sh) from a .d directory
# in sorted order. Execution is fail-fast — first failure stops all
# subsequent scripts.

run_scripts() {
	local script_dir="$1"

	if [[ ! -d "$script_dir" ]]; then
		echo "⚠ Script directory not found: $script_dir"
		return 0
	fi

	local scripts=()
	while IFS= read -r -d '' script; do
		scripts+=("$script")
	done < <(find "$script_dir" -maxdepth 1 -name '*.sh' ! -name '*.skip.sh' -print0 | sort -z)

	if [[ ${#scripts[@]} -eq 0 ]]; then
		echo "ℹ No scripts found in $script_dir"
		return 0
	fi

	for script in "${scripts[@]}"; do
		echo "▶ Running $(basename "$script")..."
		bash "$script"
		echo "✔ $(basename "$script") completed"
	done
}
