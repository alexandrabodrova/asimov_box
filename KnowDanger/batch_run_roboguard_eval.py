import subprocess
import itertools

agents = ["knowno", "roboguard", "knowdanger"]
attacks = [
    "direct_prompting",
    "template_based",
    "robopair_generated",
    "black_box",
    "gray_box_wm",
    "gray_box_gr",
    "white_box"
]

# RoboGuard experimental configuration assumptions
use_threshold = "--use_threshold"
use_rewriter = "--use_rewriter"
human_override = "--human_override"
threshold_value = "--threshold=1.2"

print("\n=== Batch Evaluation: KnowDanger vs Baselines under RoboGuard Attack Scenarios ===\n")

# Combinations of test configurations based on RoboGuard methodology
for agent, attack in itertools.product(agents, attacks):
    cmd = [
        "python", "knowdanger_vs_baselines.py",
        f"--agent={agent}",
        f"--attack={attack}"
    ]

    if agent == "knowdanger":
        cmd += [use_rewriter, use_threshold, threshold_value, human_override]

    print(f"\n[Running] Agent: {agent} | Attack: {attack}")
    subprocess.run(cmd)

print("\n=== Batch evaluation completed ===")
