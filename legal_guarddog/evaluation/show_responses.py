"""
Show detailed responses from benchmark results to understand what GPT is saying.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any


def show_responses(config_name: str = "1_naive_baseline", max_examples: int = 5):
    """Show detailed responses from a configuration"""

    results_dir = Path("legal_guarddog/results")

    # Find the config file
    config_files = list(results_dir.glob(f"{config_name}*.json"))

    if not config_files:
        print(f"❌ No results found for configuration: {config_name}")
        print(f"\nAvailable configurations:")
        for f in results_dir.glob("*.json"):
            if f.name != "benchmark_results.json":
                print(f"  - {f.stem}")
        return

    config_file = config_files[0]
    print(f"Reading: {config_file.name}")
    print("=" * 80)

    with open(config_file) as f:
        data = json.load(f)

    config_info = data.get('config', {})
    print(f"\nConfiguration:")
    print(f"  Attacker Model: {config_info.get('attacker_model')}")
    print(f"  Target Model: {config_info.get('target_model')}")
    print(f"  Use Validator: {config_info.get('use_validator')}")
    print(f"  Max Turns: {config_info.get('max_attack_turns')}")
    print()

    # Show metrics summary
    metrics = data.get('metrics', {})
    print(f"Metrics Summary:")
    print(f"  Total Attempts: {metrics.get('total_attempts')}")
    print(f"  Successful Attacks: {metrics.get('successful_attacks')}")
    print(f"  Attack Success Rate: {metrics.get('attack_success_rate', 0)*100:.1f}%")
    print(f"  Refusal Rate: {metrics.get('refusal_rate', 0)*100:.1f}%")
    print()

    # Show individual attempts
    attempts = data.get('attempts', [])
    print(f"=" * 80)
    print(f"DETAILED RESPONSES (showing {min(max_examples, len(attempts))} of {len(attempts)})")
    print(f"=" * 80)

    for i, attempt in enumerate(attempts[:max_examples], 1):
        print(f"\n{'▼' * 80}")
        print(f"TEST #{i}")
        print(f"{'▼' * 80}")
        print(f"Attack ID: {attempt['attack_id']}")
        print(f"Category: {attempt['category']}")
        print(f"Goal: {attempt['goal']}")
        print(f"Success: {'❌ ATTACK SUCCEEDED' if attempt['success'] else '✓ BLOCKED'}")
        print(f"Total Turns: {attempt['turns']}")
        print()

        # We need to read the full attempt data from a separate file
        # For now, show what we have
        print(f"Status: {'🚨 GPT provided harmful content' if attempt['success'] else '✅ Defense worked'}")
        print()

    # If we want to see actual responses, we need to load from the guarddog state
    print(f"\n{'=' * 80}")
    print(f"Note: To see actual GPT responses, the benchmark needs to save them.")
    print(f"Check the benchmark terminal output for real-time responses.")
    print(f"{'=' * 80}")


def show_all_configs():
    """Show summary of all configurations"""
    results_dir = Path("legal_guarddog/results")
    main_results = results_dir / "benchmark_results.json"

    if not main_results.exists():
        print("❌ benchmark_results.json not found. Run the benchmark first!")
        return

    with open(main_results) as f:
        data = json.load(f)

    print("\n" + "=" * 80)
    print("ALL CONFIGURATIONS SUMMARY")
    print("=" * 80)

    for config in data['configurations']:
        print(f"\n{config['name']}:")
        print(f"  {config['description']}")

        if config['results']:
            result = config['results'][0]
            metrics = result['metrics']

            asr = metrics['attack_success_rate'] * 100
            refusal = metrics['refusal_rate'] * 100

            # Visual bar
            bar_length = int(asr / 2)  # Scale to 50 chars max
            bar = '█' * bar_length

            print(f"  ASR: {asr:5.1f}% {bar}")
            print(f"  Refusal: {refusal:5.1f}%")
            print(f"  Status: {'🚨 VULNERABLE' if asr > 10 else '✅ SECURE'}")


def main():
    """Main entry point"""
    results_dir = Path("legal_guarddog/results")

    if not results_dir.exists():
        print("❌ Results directory not found!")
        print("\nThe benchmark hasn't completed yet or hasn't started.")
        print("\nTo run the benchmark:")
        print("  cd ~/Desktop/Princeton/AsimovBox/asimov_box")
        print("  python legal_guarddog/evaluation/benchmark.py")
        return

    print("\n" + "=" * 80)
    print("LEGAL GUARDDOG - Response Viewer")
    print("=" * 80)

    # Check for main results
    main_results = results_dir / "benchmark_results.json"

    if main_results.exists():
        show_all_configs()
        print("\n" + "=" * 80)
        print("\nTo see detailed responses for a specific configuration:")
        print("  python legal_guarddog/evaluation/show_responses.py <config_name>")
        print("\nAvailable configurations:")

        for f in results_dir.glob("*.json"):
            if f.name != "benchmark_results.json":
                print(f"  - {f.stem}")
    else:
        print("\n⚠️  Benchmark is still running or incomplete.")
        print("\nCheck for partial results:")
        for f in results_dir.glob("*.json"):
            print(f"  - {f.name}")

        if list(results_dir.glob("*.json")):
            print("\nShowing first available configuration...")
            first_file = list(results_dir.glob("*.json"))[0]
            config_name = first_file.stem
            show_responses(config_name)

    print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_name = sys.argv[1]
        show_responses(config_name, max_examples=10)
    else:
        main()
