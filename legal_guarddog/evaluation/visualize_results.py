"""
Enhanced visualization for Legal Guarddog benchmark results.

Creates comprehensive visualizations showing ablation study results.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from typing import Dict, List, Any


def load_benchmark_results(results_dir: str = "legal_guarddog/results") -> Dict[str, Any]:
    """Load benchmark results from JSON file"""
    results_path = Path(results_dir) / "benchmark_results.json"

    if not results_path.exists():
        raise FileNotFoundError(f"Results not found at {results_path}")

    with open(results_path, 'r') as f:
        return json.load(f)


def create_ablation_comparison_plot(results: Dict[str, Any], output_dir: str):
    """Create comprehensive 4-tier ablation comparison plot"""

    # Extract data
    configs = results["configurations"]
    config_names = []
    asrs = []
    refusal_rates = []
    colors_list = []

    # Color scheme for ablation tiers
    tier_colors = {
        "1_naive_baseline": "#d62728",  # Red (worst)
        "2_adversarial_baseline": "#ff7f0e",  # Orange (bad)
        "3_generic_safety": "#2ca02c",  # Green (better)
        "4_full_legal_guarddog": "#1f77b4"  # Blue (best)
    }

    for config in configs:
        for result in config["results"]:
            # Short label for plot
            tier_num = config["name"][0]  # Get "1", "2", "3", or "4"
            tier_names = {
                "1": "Naive\nBaseline",
                "2": "Adversarial\nBaseline",
                "3": "Generic\nSafety",
                "4": "Legal\nGuarddog"
            }
            config_names.append(tier_names.get(tier_num, config["name"]))

            metrics = result["metrics"]
            asrs.append(metrics["attack_success_rate"] * 100)
            refusal_rates.append(metrics["refusal_rate"] * 100)
            colors_list.append(tier_colors.get(config["name"], "#gray"))

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle('Legal Guarddog: 4-Tier Ablation Study Results',
                 fontsize=18, fontweight='bold', y=0.98)

    # 1. Attack Success Rate (main metric)
    ax1 = fig.add_subplot(gs[0, :])
    bars1 = ax1.bar(config_names, asrs, color=colors_list, alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Attack Success Rate (%)', fontsize=14, fontweight='bold')
    ax1.set_title('Attack Success Rate by Configuration (Lower is Better)',
                  fontsize=14, fontweight='bold')
    ax1.axhline(y=10, color='green', linestyle='--', linewidth=2, label='Target: <10%', alpha=0.7)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.legend(fontsize=11)
    ax1.set_ylim(0, max(asrs) * 1.2 if asrs else 100)

    # Add value labels on bars
    for bar, asr in zip(bars1, asrs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{asr:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add tier annotations
    for i, (bar, name) in enumerate(zip(bars1, config_names)):
        tier_descriptions = {
            "Naive\nBaseline": "Direct prompts\nNo defense",
            "Adversarial\nBaseline": "RoboPAIR attacks\nGPT guardrails only",
            "Generic\nSafety": "RoboPAIR + Generic\nSafety Validator",
            "Legal\nGuarddog": "RoboPAIR + Legal\nJudge + Validator"
        }
        desc = tier_descriptions.get(name, "")
        ax1.text(bar.get_x() + bar.get_width()/2., -max(asrs) * 0.15,
                desc, ha='center', va='top', fontsize=9, style='italic')

    # 2. Refusal Rate
    ax2 = fig.add_subplot(gs[1, 0])
    bars2 = ax2.bar(config_names, refusal_rates, color=colors_list, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Refusal Rate (%)', fontsize=12)
    ax2.set_title('Refusal Rate', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for bar, rate in zip(bars2, refusal_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)

    # 3. Defense Effectiveness Score (100 - ASR)
    ax3 = fig.add_subplot(gs[1, 1])
    effectiveness = [100 - asr for asr in asrs]
    bars3 = ax3.bar(config_names, effectiveness, color=colors_list, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Defense Effectiveness (%)', fontsize=12)
    ax3.set_title('Defense Effectiveness (Higher is Better)', fontsize=12, fontweight='bold')
    ax3.axhline(y=90, color='green', linestyle='--', linewidth=2, label='Target: >90%', alpha=0.7)
    ax3.grid(axis='y', alpha=0.3)
    ax3.legend(fontsize=10)
    for bar, eff in zip(bars3, effectiveness):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{eff:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 4. Ablation Ladder Visualization
    ax4 = fig.add_subplot(gs[2, :])
    ax4.set_xlim(0, 5)
    ax4.set_ylim(0, 100)
    ax4.set_xlabel('Ablation Tier', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Attack Success Rate (%)', fontsize=14, fontweight='bold')
    ax4.set_title('Ablation Ladder: Component Contributions', fontsize=14, fontweight='bold')

    # Plot line connecting tiers
    x_pos = [1, 2, 3, 4]
    ax4.plot(x_pos, asrs, 'o-', linewidth=3, markersize=15, color='darkblue',
             markerfacecolor='lightblue', markeredgewidth=2, markeredgecolor='darkblue')

    # Add shaded regions showing improvement
    for i in range(len(x_pos) - 1):
        ax4.fill_between([x_pos[i], x_pos[i+1]], [asrs[i], asrs[i+1]],
                        alpha=0.2, color='green', label='Improvement' if i == 0 else '')

    # Add annotations for each tier
    tier_labels = {
        1: "Tier 1:\nNaive\nBaseline",
        2: "Tier 2:\nAdversarial\nBaseline",
        3: "Tier 3:\nGeneric\nSafety",
        4: "Tier 4:\nFull\nSystem"
    }

    for x, asr, name in zip(x_pos, asrs, config_names):
        # Add value label
        ax4.text(x, asr + 5, f'{asr:.1f}%', ha='center', va='bottom',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

        # Add tier label
        tier_num = int(x)
        ax4.text(x, -10, tier_labels[tier_num], ha='center', va='top',
                fontsize=10, style='italic')

    # Add horizontal reference lines
    ax4.axhline(y=50, color='red', linestyle=':', linewidth=1, alpha=0.5, label='50% (Coin flip)')
    ax4.axhline(y=10, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Target: <10%')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    ax4.legend(fontsize=10, loc='upper right')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f"Tier {i}" for i in x_pos])

    # Save plot
    output_path = Path(output_dir) / "ablation_study_comprehensive.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Comprehensive visualization saved to: {output_path}")

    return fig


def create_category_breakdown(results: Dict[str, Any], output_dir: str):
    """Create breakdown by risk category"""

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Attack Success Rate by Risk Category', fontsize=16, fontweight='bold')

    categories = ["dual_use", "copyright", "defamation"]
    category_titles = {
        "dual_use": "Dual-Use / Public Safety",
        "copyright": "Copyright Infringement",
        "defamation": "Privacy / Defamation"
    }

    for idx, category in enumerate(categories):
        ax = axes[idx]
        ax.set_title(category_titles[category], fontsize=14, fontweight='bold')
        ax.set_ylabel('Attack Success Rate (%)', fontsize=12)
        ax.set_xlabel('Configuration', fontsize=12)

        config_names = []
        asrs_category = []
        colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']

        for i, config in enumerate(results["configurations"]):
            config_names.append(f"Tier {i+1}")

            # Get ASR for this category
            by_cat = config["results"][0]["metrics"].get("by_category", {})
            if category in by_cat:
                asrs_category.append(by_cat[category]["asr"] * 100)
            else:
                asrs_category.append(0)

        bars = ax.bar(config_names, asrs_category, color=colors, alpha=0.8, edgecolor='black')
        ax.axhline(y=10, color='green', linestyle='--', label='Target: <10%')
        ax.grid(axis='y', alpha=0.3)
        ax.legend()

        # Add value labels
        for bar, asr in zip(bars, asrs_category):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{asr:.0f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()

    output_path = Path(output_dir) / "category_breakdown.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Category breakdown saved to: {output_path}")


def generate_summary_table(results: Dict[str, Any], output_dir: str):
    """Generate markdown summary table"""

    table = "# Legal Guarddog Ablation Study Results\n\n"
    table += f"**Date:** {results['timestamp']}\n\n"
    table += f"**Test Cases:** {results['test_cases']['adversarial']} adversarial, "
    table += f"{results['test_cases']['benign']} benign\n\n"

    table += "## Summary Table\n\n"
    table += "| Tier | Configuration | ASR ↓ | Refusal Rate | Defense Effectiveness ↑ | Key Finding |\n"
    table += "|------|---------------|-------|--------------|------------------------|-------------|\n"

    findings = {
        "1_naive_baseline": "GPT blocks simple attacks",
        "2_adversarial_baseline": "**Jailbreaks bypass GPT**",
        "3_generic_safety": "Generic safety helps but incomplete",
        "4_full_legal_guarddog": "**Legal policies achieve near-perfect defense**"
    }

    for i, config in enumerate(results["configurations"], 1):
        result = config["results"][0]
        metrics = result["metrics"]
        asr = metrics["attack_success_rate"] * 100
        refusal = metrics["refusal_rate"] * 100
        effectiveness = 100 - asr

        table += f"| {i} | {config['description'][:40]} | "
        table += f"{asr:.1f}% | {refusal:.1f}% | {effectiveness:.1f}% | "
        table += f"{findings.get(config['name'], 'N/A')} |\n"

    table += "\n## Key Insights\n\n"
    table += "1. **Naive attacks** are well-defended by GPT's built-in guardrails\n"
    table += "2. **Adversarial attacks (RoboPAIR)** expose significant vulnerabilities\n"
    table += "3. **Generic safety** helps but misses legal-specific requirements\n"
    table += "4. **Legal Guarddog** achieves robust defense through legal-specific policies\n"

    output_path = Path(output_dir) / "results_summary.md"
    with open(output_path, 'w') as f:
        f.write(table)

    print(f"✓ Summary table saved to: {output_path}")
    print("\n" + table)


def main():
    """Generate all visualizations"""
    print("\n" + "="*70)
    print("LEGAL GUARDDOG - Results Visualization")
    print("="*70 + "\n")

    # Load results
    print("Loading benchmark results...")
    results = load_benchmark_results()

    output_dir = "legal_guarddog/results"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    print("\nGenerating visualizations...\n")

    create_ablation_comparison_plot(results, output_dir)
    create_category_breakdown(results, output_dir)
    generate_summary_table(results, output_dir)

    print("\n" + "="*70)
    print("✓ All visualizations complete!")
    print("="*70)
    print(f"\nFiles generated in: {output_dir}/")
    print("  - ablation_study_comprehensive.png")
    print("  - category_breakdown.png")
    print("  - results_summary.md")
    print("  - benchmark_results.json")
    print()


if __name__ == "__main__":
    main()
