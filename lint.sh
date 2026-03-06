#!/usr/bin/env bash
# Lint the codebase with flake8, black, and isort.
# Usage:
#   ./lint.sh           # check only (no changes)
#   ./lint.sh --fix     # auto-fix with black and isort where possible
#
# Override paths: LINT_PATHS="scripts src" ./lint.sh
# Use 100-char lines: LINE_LENGTH=100 ./lint.sh --fix

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Paths to lint (default: scripts)
PATHS="${LINT_PATHS:-scripts}"
# Line length (default 120; Black fixes E501 by reflowing to this length)
LINE_LENGTH="${LINE_LENGTH:-120}"

FIX_MODE=false
if [[ "${1:-}" == "--fix" ]]; then
  FIX_MODE=true
fi

echo "=== Linting (PATHS=${PATHS}, FIX=${FIX_MODE}) ==="

# --- isort (import sorting) ---
echo ""
echo "--- isort ---"
if $FIX_MODE; then
  isort $PATHS
else
  isort $PATHS --check-only --diff || true
fi
ISORT_EXIT=$?

# --- black (formatting; fixes E501 by reflowing long lines) ---
echo ""
echo "--- black ---"
if $FIX_MODE; then
  black $PATHS --line-length "$LINE_LENGTH"
else
  black $PATHS --line-length "$LINE_LENGTH" --check --diff || true
fi
BLACK_EXIT=$?

# --- flake8 (ignores and options from .flake8; line-length overridable here) ---
echo ""
echo "--- flake8 ---"
flake8 $PATHS --max-line-length="$LINE_LENGTH" || true
FLAKE8_EXIT=$?

# Summary
echo ""
echo "=== Summary ==="
echo "isort:  $([ $ISORT_EXIT -eq 0 ] && echo 'OK' || echo 'FAIL')"
echo "black:  $([ $BLACK_EXIT -eq 0 ] && echo 'OK' || echo 'FAIL')"
echo "flake8: $([ $FLAKE8_EXIT -eq 0 ] && echo 'OK' || echo 'FAIL')"

exit $((ISORT_EXIT + BLACK_EXIT + FLAKE8_EXIT))
