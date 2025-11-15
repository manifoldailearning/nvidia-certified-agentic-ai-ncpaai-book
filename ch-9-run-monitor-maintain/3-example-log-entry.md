# Run, Monitor, and Maintain

This chapter covers monitoring, drift detection, and maintenance of AI agents in production.

## Example Log Entries

The following are example log entries showing agent response monitoring with latency, drift detection, and status information:

```
2025-11-04T09:42:11Z | agent_response | latency=0.29s | drift=0.18 | status=alert  
2025-11-04T09:45:18Z | agent_response | latency=0.22s | drift=0.10 | status=stable
```

### Log Entry Format

Each log entry contains:
- **Timestamp**: ISO 8601 format (UTC)
- **Event Type**: `agent_response`
- **Latency**: Response time in seconds
- **Drift**: Drift metric value (higher values indicate more drift)
- **Status**: Current status (`alert` or `stable`)

### Status Indicators

- **`alert`**: Drift detected above threshold - requires attention
- **`stable`**: Model behavior is within acceptable drift range

