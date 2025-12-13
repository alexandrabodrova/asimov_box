
"""
Visualization functions for baseline test results.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np


def plot_asr_comparison(naive_asr: Optional[float] = None,
                       pair_asr: Optional[float] = None,
                       full_asr: Optional[float] = None,
                       output_dir: str = "legal_guarddog/results"):
    """
    Create simple bar chart comparing ASR across baselines.

    Args:
        naive_asr: Naive baseline ASR (optional)
        pair_asr: PAIR baseline ASR (optional)
        full_asr: Full judge baseline ASR (required)
        output_dir: Directory to save plot
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Prepare data
    baselines = []
    asrs = []
    colors = []

    if naive_asr is not None:
        baselines.append('Naive\n(Tier 1)')
        asrs.append(naive_asr)
        colors.append('#95e1d3')

    if pair_asr is not None:
        baselines.append('PAIR\n(Simple Judge)')
        asrs.append(pair_asr)
        colors.append('#feca57')

    if full_asr is not None:
        baselines.append('Full Judge\n(Policy)')
        asrs.append(full_asr)
        colors.append('#ff6b6b')

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(baselines, asrs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add percentage labels on bars
    for bar, asr in zip(bars, asrs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{asr:.1f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Formatting
    ax.set_ylabel('Attack Success Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Baseline Comparison: Attack Success Rate', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.3, linewidth=2, label='50% threshold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/asr_comparison_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ ASR comparison plot saved to: {filename}")
    return filename


def plot_turns_distribution(pair_results: Optional[List[dict]] = None,
                           full_results: Optional[List[dict]] = None,
                           output_dir: str = "legal_guarddog/results"):
    """
    Create turn distribution plot showing turns to success by category.

    Args:
        pair_results: PAIR baseline results (optional)
        full_results: Full judge baseline results (optional)
        output_dir: Directory to save plot
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Prepare data by category
    from legal_guarddog.policies.legal_policy_engine import RiskCategory

    categories = [RiskCategory.DUAL_USE, RiskCategory.COPYRIGHT, RiskCategory.DEFAMATION]
    category_labels = ['Dual-Use', 'Copyright', 'Defamation']

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Distribution of Turns to Success by Category', fontsize=16, fontweight='bold')

    for i, (category, label) in enumerate(zip(categories, category_labels)):
        ax = axes[i]

        # Collect successful attack turns for this category
        pair_turns = []
        full_turns = []

        if pair_results:
            pair_turns = [r['turns'] for r in pair_results
                         if r['category'] == category and r['success']]

        if full_results:
            full_turns = [r['turns'] for r in full_results
                         if r['category'] == category and r['success']]

        # Create histogram
        bins = np.arange(0.5, 6.5, 1)  # 1, 2, 3, 4, 5 turns

        if pair_turns:
            ax.hist(pair_turns, bins=bins, alpha=0.7, label='PAIR', color='#feca57', edgecolor='black')

        if full_turns:
            ax.hist(full_turns, bins=bins, alpha=0.7, label='Full Judge', color='#ff6b6b', edgecolor='black')

        # Formatting
        ax.set_xlabel('Turns to Success', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlim(0.5, 5.5)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        if i == 0:  # Only show legend on first plot
            ax.legend()

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/turns_distribution_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Turn distribution plot saved to: {filename}")
    return filename


def plot_individual_baseline_asr(results: List[dict], baseline_name: str,
                                 output_dir: str = "legal_guarddog/results_visualization"):
    """
    Plot ASR by category for a single baseline.

    Args:
        results: List of result dictionaries
        baseline_name: Name of baseline ("PAIR" or "Full Judge")
        output_dir: Directory to save plot
    """
    from legal_guarddog.policies.legal_policy_engine import RiskCategory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Calculate ASR by category
    categories = [RiskCategory.DUAL_USE, RiskCategory.COPYRIGHT, RiskCategory.DEFAMATION]
    category_labels = ['Dual-Use', 'Copyright', 'Defamation']
    asrs = []

    for category in categories:
        cat_results = [r for r in results if r['category'] == category]
        if cat_results:
            cat_successes = sum(1 for r in cat_results if r['success'])
            asr = (cat_successes / len(cat_results)) * 100
        else:
            asr = 0
        asrs.append(asr)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    color = '#feca57' if baseline_name == 'PAIR' else '#ff6b6b'
    bars = ax.bar(category_labels, asrs, color=color, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add percentage labels
    for bar, asr in zip(bars, asrs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{asr:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Formatting
    ax.set_ylabel('Attack Success Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'{baseline_name} Baseline: ASR by Category', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.3, linewidth=2, label='50% threshold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/{baseline_name.lower().replace(' ', '_')}_asr_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ {baseline_name} ASR plot saved to: {filename}")
    return filename


def plot_individual_baseline_attempts(results: List[dict], baseline_name: str,
                                      output_dir: str = "legal_guarddog/results_visualization"):
    """
    Plot attempt count distribution for a single baseline.

    Args:
        results: List of result dictionaries
        baseline_name: Name of baseline ("PAIR" or "Full Judge")
        output_dir: Directory to save plot
    """
    from legal_guarddog.policies.legal_policy_engine import RiskCategory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    categories = [RiskCategory.DUAL_USE, RiskCategory.COPYRIGHT, RiskCategory.DEFAMATION]
    category_labels = ['Dual-Use', 'Copyright', 'Defamation']

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'{baseline_name} Baseline: Attempt Count Distribution by Category',
                fontsize=16, fontweight='bold')

    color = '#feca57' if baseline_name == 'PAIR' else '#ff6b6b'

    for i, (category, label) in enumerate(zip(categories, category_labels)):
        ax = axes[i]

        # Collect successful attack turns for this category
        turns = [r['turns'] for r in results
                if r['category'] == category and r['success']]

        # Create histogram
        bins = np.arange(0.5, 6.5, 1)  # 1, 2, 3, 4, 5 turns

        if turns:
            ax.hist(turns, bins=bins, alpha=0.8, color=color, edgecolor='black', linewidth=1.5)

        # Formatting
        ax.set_xlabel('Attempts to Success', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlim(0.5, 5.5)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add count annotation
        if turns:
            ax.text(0.95, 0.95, f'Total: {len(turns)} successes',
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=10)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/{baseline_name.lower().replace(' ', '_')}_attempts_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ {baseline_name} attempts plot saved to: {filename}")
    return filename
