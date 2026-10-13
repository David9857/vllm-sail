#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# One Python interpreter manages argument arrays, per-file execution, and failures.
set -Eeuo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python "${ROOT_DIR}/scripts/ci/ppu_ops_runner.py" run
