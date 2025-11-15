import json
import time

def monitor_hook(query, response_time, drift_score):
    log = {
        "timestamp": time.time(),
        "query": query,
        "latency": response_time,
        "drift_score": drift_score
    }
    with open("agent_logs.jsonl", "a") as f:
        f.write(json.dumps(log) + "\n")

# Example call
monitor_hook("Summarize report", 0.21, 0.12)
