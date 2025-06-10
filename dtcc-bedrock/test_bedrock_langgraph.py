import os
import boto3
import json

from dotenv import load_dotenv
load_dotenv()

def test_bedrock_connection():
    bedrock = boto3.client(
        service_name="bedrock-runtime",
        region_name=os.getenv("AWS_REGION", "us-east-2"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
    )

    model_id = "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-3-5-sonnet-20240620-v1:0"
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [
            {"role": "user", "content": "Hello! What is the capital of France?"}
        ],
        "max_tokens": 50,
        "temperature": 0.5,
        "top_p": 1
    }

    try:
        response = bedrock.invoke_model(
            modelId=model_id,
            body=json.dumps(payload).encode("utf-8"),
            accept="application/json",
            contentType="application/json"
        )
        print("Model response:")
        print(response["body"].read().decode())
        return True
    except Exception as e:
        print("Error invoking model:", e)
        return False

if __name__ == "__main__":
    test_bedrock_connection()
