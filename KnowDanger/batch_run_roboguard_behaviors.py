import subprocess
import itertools
from RoboGuard1.prompts.base import roboguard_test_prompts

agents = ["knowno", "roboguard", "knowdanger"]
behaviors = list(roboguard_test_prompts.keys())

use_threshold = "--use_threshold"
use_rewriter = "--use_rewriter"
human_override = "--human_override"
threshold_value = "--threshold=1.2"

print("\n=== Batch Evaluation: KnowDanger vs Baselines on RoboGuard Behaviors ===\n")

for agent, behavior in itertools.product(agents, behaviors):
    cmd = [
        "python", "knowdanger_vs_baselines.py",
        f"--agent={agent}",
        f"--behavior={behavior}"
    ]

    if agent == "knowdanger":
        cmd += [use_rewriter, use_threshold, threshold_value, human_override]

    print(f"\n[Running] Agent: {agent} | Behavior: {behavior}")
    subprocess.run(cmd)

print("\n=== Batch evaluation completed ===")
