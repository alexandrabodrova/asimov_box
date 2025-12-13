"""
Save detailed adversarial test results to a text file.
"""

from datetime import datetime
from pathlib import Path
from typing import List


def save_detailed_results(results: List[dict], attempts: List, asr: float, config=None, output_dir: str = "legal_guarddog/results"):
    """
    Save detailed adversarial test results including all prompts and responses.

    Args:
        results: Summary results list
        attempts: List of AttackAttempt objects with full conversation history
        asr: Attack Success Rate
        config: Configuration object (optional, for displaying actual settings)
        output_dir: Directory to save results
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/adversarial_baseline_detailed_{timestamp}.txt"

    with open(filename, 'w', encoding='utf-8') as f:
        # Header
        f.write("="*80 + "\n")
        f.write("ADVERSARIAL BASELINE TEST - DETAILED RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Configuration: RoboPAIR-style adversarial attacks vs GPT-4\n")

        # Display actual configuration if provided
        if config:
            f.write(f"  - Attacker: {'ON' if config.use_attacker else 'OFF'} {'(generates sophisticated jailbreaks)' if config.use_attacker else ''}\n")
            f.write(f"  - Judge: {'ON' if config.use_judge else 'OFF'} {'(evaluates responses against policy rules)' if config.use_judge else '(no policy checking)'}\n")
            f.write(f"  - Validator: {'ON' if config.use_validator else 'OFF'} {'(defense layer active)' if config.use_validator else '(no defense layer)'}\n")
            f.write(f"  - Max turns per attack: {config.max_attack_turns}\n")
        else:
            # Fallback if config not provided
            f.write(f"  - Attacker: ON (generates sophisticated jailbreaks)\n")
            f.write(f"  - Judge: Unknown (config not provided)\n")
            f.write(f"  - Validator: OFF (no defense layer)\n")
            f.write(f"  - Max turns per attack: 5\n")
        f.write("\n")

        # Summary
        f.write("="*80 + "\n")
        f.write("SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Total attacks: {len(results)}\n")
        f.write(f"Successful jailbreaks: {sum(1 for r in results if r['success'])}\n")
        f.write(f"Attack Success Rate (ASR): {asr:.1f}%\n")
        f.write("\n")

        f.write("Per-category results:\n")
        for r in results:
            status = "❌ JAILBROKEN" if r['success'] else "✅ BLOCKED"
            f.write(f"  [{status}] {r['category']}: {r['goal']} ({r['turns']} turns)\n")
        f.write("\n\n")

        # Detailed attack-by-attack breakdown
        for i, (result, attempt) in enumerate(zip(results, attempts), 1):
            f.write("="*80 + "\n")
            f.write(f"ATTACK {i}/{len(results)}: {result['category'].upper()}\n")
            f.write("="*80 + "\n")
            f.write(f"Goal: {result['goal']}\n")
            f.write(f"Success: {'❌ YES - Jailbreak succeeded' if result['success'] else '✅ NO - GPT blocked all attempts'}\n")
            f.write(f"Turns: {result['turns']}/5\n")
            f.write("\n")

            # Turn-by-turn breakdown
            for j, turn in enumerate(attempt.turns, 1):
                f.write("-"*80 + "\n")
                f.write(f"TURN {j}/{len(attempt.turns)}\n")
                f.write("-"*80 + "\n")
                f.write("\n")

                f.write("ATTACKER PROMPT:\n")
                f.write("-"*80 + "\n")
                f.write(turn.attacker_prompt)
                f.write("\n\n")

                f.write("GPT-4 RESPONSE:\n")
                f.write("-"*80 + "\n")
                f.write(turn.target_response)
                f.write("\n\n")

                f.write("ANALYSIS:\n")
                f.write("-"*80 + "\n")
                refusal = turn.judge_verdict.is_compliant if hasattr(turn.judge_verdict, 'is_compliant') else None
                risk_score = turn.judge_verdict.risk_score if hasattr(turn.judge_verdict, 'risk_score') else None

                if refusal is not None:
                    f.write(f"  GPT Refused: {refusal}\n")
                if risk_score is not None:
                    f.write(f"  Risk Score: {risk_score:.2f}\n")

                f.write(f"  Validator Action: {turn.validator_action.value}\n")
                f.write(f"  Response Length: {len(turn.target_response)} characters\n")
                f.write("\n")

                # If this turn led to success, mark it
                if j == len(attempt.turns) and attempt.success:
                    f.write("🚨 ATTACK SUCCEEDED ON THIS TURN\n")
                    f.write("   GPT provided harmful content that was not refused.\n")
                    f.write("\n")

            # Final outcome for this attack
            f.write("-"*80 + "\n")
            f.write("FINAL OUTCOME:\n")
            f.write("-"*80 + "\n")
            if result['success']:
                f.write(f"❌ JAILBREAK SUCCEEDED after {result['turns']} turn(s)\n")
                f.write(f"   GPT-4's guardrails were bypassed.\n")
            else:
                f.write(f"✅ ALL ATTACKS BLOCKED\n")
                f.write(f"   GPT-4 refused all {result['turns']} jailbreak attempts.\n")
            f.write("\n\n\n")

        # Footer
        f.write("="*80 + "\n")
        f.write("END OF DETAILED RESULTS\n")
        f.write("="*80 + "\n")

    print(f"\n✓ Detailed results saved to: {filename}")
    return filename
