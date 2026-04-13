# DevContainer Definition

## Overview

Defines the project-specific DevContainer requirements. Populate the sections below with required additions for this repository.

## Generated Notice

When generating the `devcontainer.json`, include a top-of-file JSONC notice that points to this standard and instructs contributors to update the spec first and regenerate.

```jsonc
// Generated from definition.md. Do not edit directly.
```

---

## Provider

Azure DevOps is the provider for this repository.

## Languages

- **Python 3.12** — ML pipeline scripts, Azure ML SDK, data processing

## Ports

| Port | Purpose |
| ---- | ------- |
|      |         |

- None. No port forwarding at this point.

## Features

Required additional features beyond the DevContainer defaults:

- **Azure CLI** with `azure-devops` and `ml` extensions — for Azure ML workspace interaction

Excluded inherited features:

- `ghcr.io/devcontainers/features/git-lfs:1` — not required for this repository

## Mounts

No additional mounts beyond the DevContainer defaults.

## Extensions

- **ms-python.python** — Python IntelliSense, debugging, linting
- **charliermarsh.ruff** — Ruff linter and formatter

---

## Post-Create Hooks

- Install Python 3.12 and build dependencies
- Install ML pipeline dependencies from `ml-pipeline/requirements.txt`

## Post-Start Hooks

None.
