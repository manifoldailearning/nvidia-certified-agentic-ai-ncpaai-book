import json
import os
from datetime import datetime, timezone

BASELINE_FILE = "drift_baseline.json"

def save_baseline(vector):
    """Persist the baseline vector to a local JSON file."""
    with open(BASELINE_FILE, "w") as f:
        json.dump({"baseline": vector}, f)

def load_baseline():
    """Load baseline vector if it exists."""
    if not os.path.exists(BASELINE_FILE):
        return None
    with open(BASELINE_FILE, "r") as f:
        data = json.load(f)
        return data.get("baseline")

def detect_drift(baseline, current, threshold=0.15, metric="feature_importance_drift"):
    """
    Simple L1 drift detection.
    baseline/current are vectors like [0.1, 0.5, 0.4].
    Returns a dictionary with drift detection results in standardized format.
    """
    if baseline is None:
        print("No baseline found. Initializing with current vector.")
        save_baseline(current)
        return {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "metric": metric,
            "baseline_distribution": current,
            "current_distribution": current,
            "drift_detected": False
        }

    diff = sum(abs(b - c) for b, c in zip(baseline, current))
    drift_detected = diff > threshold

    if drift_detected:
        print(f"Drift detected! change={diff:.3f} > threshold={threshold}")
    else:
        print(f"Model stable. change={diff:.3f}")

    return {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "metric": metric,
        "baseline_distribution": baseline,
        "current_distribution": current,
        "drift_detected": drift_detected
    }


# ---------------------------
# Example Usage
# ---------------------------

# Baseline feature vector (e.g., feature importance or embedding stats)
initial_vector = [0.1, 0.5, 0.4]
save_baseline(initial_vector)

# New vector from current model behavior
current_vector = [0.2, 0.7, 0.1]

baseline = load_baseline()
result = detect_drift(baseline, current_vector)

# Output in standardized JSON format
print("\nDrift Detection Result:")
print(json.dumps(result, indent=2))
