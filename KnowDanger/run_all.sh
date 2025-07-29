#!/bin/bash

# Full Evaluation: KnowDanger Box + Baselines across RoboGuard behaviors
#export PYTHONPATH="/Users/abodrova/Desktop/Princeton/AsimovBox:/Users/abodrova/Desktop/Princeton/AsimovBox/RoboGuard/src:$PYTHONPATH"

echo "Running batch evaluation of KnowDanger and all baselines..."
python batch_run_roboguard_behaviors.py

echo "Summarizing results..."
python summarize_roboguard_behavior_results.py

echo "Zipping result files..."
zip -r roboguard_results.zip roboguard_eval_*.csv roboguard_behavior_summary.csv roboguard_behavior_asr_plot.png

echo "Done. Results saved in roboguard_results.zip"
