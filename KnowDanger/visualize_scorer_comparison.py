import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
df = pd.read_csv("scorer_comparison.csv")

# Convert 'logprob_gap' column to float
df["logprob_gap"] = pd.to_numeric(df["logprob_gap"], errors="coerce")

# Optional: Simulated correctness labels (in real use, this should come from ground truth)
df["is_correct"] = df["top_token"].apply(lambda x: x.strip().upper() == "A")

# Plot histogram of logprob gaps per scorer
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x="logprob_gap", hue="scorer", bins=10, kde=True, palette="Set2")
plt.axvline(1.0, color='red', linestyle='--', label='Threshold = 1.0')
plt.title("Logprob Gap Distribution by Scorer")
plt.xlabel("Logprob Gap")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("logprob_gap_histogram.png")
plt.show()

# Reliability curve (calibration-style)
bins = pd.qcut(df["logprob_gap"], q=5, duplicates='drop')
calib = df.groupby(bins).agg({"is_correct": "mean", "logprob_gap": "mean"}).reset_index()

plt.figure(figsize=(8, 6))
plt.plot(calib["logprob_gap"], calib["is_correct"], marker='o', label="Empirical Accuracy")
plt.plot([0, df["logprob_gap"].max()], [0, 1], linestyle='--', color='gray', label="Perfect Calibration")
plt.xlabel("Logprob Gap")
plt.ylabel("Accuracy")
plt.title("Reliability Curve (Calibration Plot)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("logprob_calibration_curve.png")
plt.show()

print("Visualizations saved: logprob_gap_histogram.png, logprob_gap_vs_time.png, logprob_calibration_curve.png")
