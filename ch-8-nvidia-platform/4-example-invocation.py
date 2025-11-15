# snippet from the book - for reference only
import json
import boto3
sagemaker_runtime = boto3.client('sagemaker-runtime')

response = sagemaker_runtime.invoke_endpoint(
  EndpointName='agentic-triton-endpoint',
  Body=json.dumps({'text': 'Generate summary for AI trends'})
)
print(response['Body'].read())
