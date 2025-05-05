# === Dataset Format Example (CSV) ===
# Each row corresponds to one prompt-action case from LLM
# Fields: logprob_gap, top_token, is_top_token_yes, num_choices, entropy, instruction_length, contains_spatial, scene_complexity, is_correct

"""logprob_gap,top_token,is_top_token_yes,num_choices,entropy,instruction_length,contains_spatial,scene_complexity,is_correct
1.5,A,1,4,0.82,13,1,6,1
0.4,B,0,4,1.93,12,1,7,0
1.8,A,1,4,0.64,10,0,5,1
0.6,D,0,4,1.88,15,1,6,0"""


# === Classifier Training & Evaluation ===
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

# Load and preprocess data
df = pd.read_csv("logprob_classification_data.csv")
X = df.drop(columns=["is_correct", "top_token"])
y = df["is_correct"]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Predict & evaluate
y_pred = clf.predict(X_test)
y_probs = clf.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_probs))

# Reliability plot (probability bins vs. true accuracy)
def plot_reliability_curve(probs, labels):
    bins = np.linspace(0, 1, 11)
    binids = np.digitize(probs, bins) - 1
    accuracies = [labels[binids == i].mean() if any(binids == i) else np.nan for i in range(len(bins)-1)]
    plt.plot(bins[:-1] + 0.05, accuracies, marker='o')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("Predicted Probability")
    plt.ylabel("Empirical Accuracy")
    plt.title("Reliability Plot")
    plt.grid(True)
    plt.savefig("reliability_plot.png")
    plt.show()

plot_reliability_curve(y_probs, y_test)

# Save model
import joblib
joblib.dump(clf, "safety_classifier.joblib")


# === AsimovBox integration ===
from joblib import load

class LearnedSafetyCertifier:
    def __init__(self, clf_path: str, feature_extractor: callable, threshold: float = 0.9):
        self.model = load(clf_path)
        self.extract_features = feature_extractor
        self.threshold = threshold

    def certify(self, prompt: str, logprobs: dict, metadata: dict) -> bool:
        x = self.extract_features(prompt, logprobs, metadata)
        p_safe = self.model.predict_proba([x])[0][1]
        print(f"[LearnedCertifier] p_safe = {p_safe:.2f}")
        return p_safe > self.threshold

# Example placeholder for feature extraction
# Should match this to the features you trained with
def extract_features(prompt, logprobs, metadata):
    sorted_probs = sorted(logprobs.items(), key=lambda x: x[1], reverse=True)
    gap = sorted_probs[0][1] - sorted_probs[1][1]
    entropy = -sum(p * np.log(p + 1e-6) for _, p in logprobs.items())
    return [
        gap,
        1 if sorted_probs[0][0] == 'A' else 0,
        metadata.get("num_choices", 4),
        entropy,
        len(prompt.split()),
        int(any(word in prompt.lower() for word in ["left", "right", "above", "below"])),
        metadata.get("scene_complexity", 6)
    ]
