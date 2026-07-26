#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/run_adapt_tier.sh" smoke
"${SCRIPT_DIR}/run_dist_tier.sh" quick
