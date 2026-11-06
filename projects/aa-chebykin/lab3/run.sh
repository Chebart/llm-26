#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PATH="$PWD/data/runtime/node-linux/bin:$PATH"

if [[ ! -x .venv-wsl/bin/python ]]; then
    printf '%s\n' 'WSL Python environment not found. Follow the WSL setup in README.' >&2
    exit 1
fi

exec .venv-wsl/bin/python -m core.cli run "$@"
