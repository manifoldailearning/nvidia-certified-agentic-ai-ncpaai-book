from prometheus_client import Counter, Histogram, start_http_server
import time

REQUEST_COUNT = Counter('agent_requests_total', 'Total requests processed')
RESPONSE_LATENCY = Histogram('agent_response_latency_seconds',
                             'Response latency in seconds')

start_http_server(8000)

def agent_respond(query):
    start = time.time()
    time.sleep(0.2)  # Simulated inference time
    RESPONSE_LATENCY.observe(time.time() - start)
    REQUEST_COUNT.inc()
    return "Response generated"

while True:
    agent_respond("What is AI?")
