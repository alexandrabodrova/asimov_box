#!/bin/bash
# Check Legal Guarddog benchmark results

echo "==================================================================="
echo "LEGAL GUARDDOG RESULTS CHECKER"
echo "==================================================================="
echo ""

RESULTS_DIR="legal_guarddog/results"

# Check if results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "❌ Results directory not found yet: $RESULTS_DIR"
    echo ""
    echo "Possible reasons:"
    echo "  1. Benchmark is still running"
    echo "  2. Benchmark hasn't started yet"
    echo "  3. Benchmark encountered an error"
    echo ""
    echo "To check if benchmark is running:"
    echo "  ps aux | grep benchmark.py"
    echo ""
    exit 1
fi

echo "✓ Results directory found: $RESULTS_DIR"
echo ""

# List all result files
echo "Files in results directory:"
ls -lh "$RESULTS_DIR"
echo ""

# Check for main results file
if [ -f "$RESULTS_DIR/benchmark_results.json" ]; then
    echo "✓ Main results file found!"
    echo ""

    # Extract summary
    python3 << 'PYTHON_SCRIPT'
import json
from pathlib import Path

results_file = Path("legal_guarddog/results/benchmark_results.json")
with open(results_file) as f:
    data = json.load(f)

print("="*60)
print("BENCHMARK SUMMARY")
print("="*60)
print(f"Timestamp: {data['timestamp']}")
print(f"Total configurations: {len(data['configurations'])}")
print()

for config in data['configurations']:
    print(f"\n{config['name']}:")
    print(f"  Description: {config['description']}")

    if config['results']:
        result = config['results'][0]
        metrics = result['metrics']
        print(f"  ASR: {metrics['attack_success_rate']*100:.1f}%")
        print(f"  Refusal Rate: {metrics['refusal_rate']*100:.1f}%")
        print(f"  Total Attempts: {metrics.get('total_attempts', 'N/A')}")
PYTHON_SCRIPT

else
    echo "⚠️  Main benchmark_results.json not found yet"
    echo ""
fi

# Check for individual config files
echo ""
echo "Individual configuration files:"
for file in "$RESULTS_DIR"/*.json; do
    if [ -f "$file" ]; then
        basename=$(basename "$file")
        echo "  - $basename"
    fi
done
echo ""

# Show sample responses if available
echo "==================================================================="
echo "SAMPLE RESPONSES (First 3 attempts)"
echo "==================================================================="

for config_file in "$RESULTS_DIR"/1_naive_baseline*.json; do
    if [ -f "$config_file" ]; then
        echo ""
        echo "From: $(basename "$config_file")"
        echo ""

        python3 << PYTHON_SCRIPT
import json
import sys

try:
    with open("$config_file") as f:
        data = json.load(f)

    for i, attempt in enumerate(data.get('attempts', [])[:3], 1):
        print(f"\n{'─'*60}")
        print(f"Test {i}: {attempt['goal']}")
        print(f"Success: {'❌ ATTACK SUCCEEDED' if attempt['success'] else '✓ BLOCKED'}")
        print(f"Turns: {attempt['turns']}")
        print(f"{'─'*60}")
except Exception as e:
    print(f"Error reading file: {e}", file=sys.stderr)
PYTHON_SCRIPT

        break
    fi
done

echo ""
echo "==================================================================="
echo "To see detailed responses, run:"
echo "  python legal_guarddog/evaluation/show_responses.py"
echo "==================================================================="
