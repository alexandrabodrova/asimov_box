#!/usr/bin/env bash
set -euo pipefail

# Resolve paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"   # AsimovBox
KNOWDANGER_DIR="$SCRIPT_DIR"                         # .../asimov_box/KnowDanger
ROBOGUARD_SRC="$PROJECT_ROOT/RoboGuard/src"

# Env needed for imports & APIs
export PYTHONPATH="$PROJECT_ROOT:$ROBOGUARD_SRC:${PYTHONPATH:-}"

# Optional: require OpenAI key only if you use --scorer gpt4
# if [[ "${1:-}" == *"--scorer"* && "${*}" == *"gpt4"* ]]; then
#   : "${OPENAI_API_KEY:?Set OPENAI_API_KEY in your shell or here}"
# fi

echo "Running batch evaluation of KnowDanger and all baselines..."
python "$KNOWDANGER_DIR/batch_run_roboguard_behaviors.py" "$@"

echo "Summarizing results..."
python "$KNOWDANGER_DIR/summarize_roboguard_behavior_results.py" --glob 'roboguard_eval_*.csv'

echo "Zipping result files..."
cd "$KNOWDANGER_DIR"
zip -r roboguard_results.zip . -i 'roboguard_eval_*.csv' 'roboguard_behavior_summary.csv' 'roboguard_behavior_asr_plot.png' || true
echo "Done. Results saved in $KNOWDANGER_DIR/roboguard_results.zip"
