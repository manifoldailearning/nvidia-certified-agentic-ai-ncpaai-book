from langsmith import Client
client = Client()

results = client.evaluate(
    dataset='agentic-ai-eval',
    run_factory='production-agent',
    evaluators=['accuracy', 'coherence', 'safety']
)
print(results.summary())
