import csv
import matplotlib.pyplot as plt
from datetime import datetime


def load_logprob_gaps(log_file):
    timestamps, gaps = [], []
    with open(log_file, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                timestamp = datetime.fromisoformat(row['timestamp'])
                gap = float(row['logprob_gap'])
                timestamps.append(timestamp)
                gaps.append(gap)
            except Exception:
                continue
    return timestamps, gaps


def plot_histogram(gaps, threshold):
    plt.figure(figsize=(8, 5))
    plt.hist(gaps, bins=20, color='steelblue', edgecolor='black')
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
    plt.title("Distribution of Logprob Gaps")
    plt.xlabel("Logprob Gap")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("logprob_gap_histogram.png")
    plt.savefig("logprob_gap_histogram.jpg")
    plt.show()


def plot_time_series(timestamps, gaps, threshold):
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, gaps, marker='o', linestyle='-', color='darkgreen', label='Logprob Gap')
    plt.axhline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
    plt.title("Logprob Gap Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Logprob Gap")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("logprob_gap_timeseries.png")
    plt.savefig("logprob_gap_timeseries.jpg")
    plt.show()


if __name__ == "__main__":
    log_file = "logprob_gaps.csv"
    threshold = 1.0
    timestamps, gaps = load_logprob_gaps(log_file)

    if not gaps:
        print("No logprob data found.")
    else:
        plot_histogram(gaps, threshold)
        plot_time_series(timestamps, gaps, threshold)
