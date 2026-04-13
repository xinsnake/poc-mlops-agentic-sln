#!/bin/bash
set -euo pipefail

echo "--- Installing Python and ML pipeline dependencies ---"

# Install Python and build dependencies
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-pip python3.12-venv python3-dev build-essential libssl-dev libffi-dev

# Make python3 available as python
sudo ln -sf /usr/bin/python3 /usr/local/bin/python
sudo ln -sf /usr/bin/pip3 /usr/local/bin/pip

# Install ML pipeline dependencies
WORKSPACE_DIR="${containerWorkspaceFolder:-/workspaces/poc-mlops-agentic-sln}"
REQ_FILE="$WORKSPACE_DIR/ml-pipeline/requirements.txt"

if [ -f "$REQ_FILE" ]; then
	# snowflake-connector needs pandas extra for write_pandas support
	pip3 install --break-system-packages "snowflake-connector-python[pandas]" --prefer-binary
	pip3 install --break-system-packages -r "$REQ_FILE" --prefer-binary
	echo "--- ML pipeline dependencies installed ---"
else
	echo "--- No requirements.txt found at $REQ_FILE, skipping ---"
fi
