#!/bin/bash
set -euo pipefail

# Configure git safe.directory for the container workspace.
# Prevents git from refusing to operate due to ownership mismatches
# between host and container users.
git config --global --add safe.directory "${containerWorkspaceFolder:-/workspaces}"
