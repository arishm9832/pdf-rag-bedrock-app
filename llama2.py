import boto3
import json

# Prompt data
prompt_data = "Act as Shakespeare and write a poem on Generative AI"

# Bedrock client with explicit region
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2"  # Replace with your actual region
)

# Payload for LLaMA3
payload = {
    "prompt": f"[INST] {prompt_data} [/INST]",
    "max_gen_len": 512,
    "temperature": 0.5,
    "top_p": 0.9
}

# Serialize payload
body = json.dumps(payload)

# Model ID
model_id = "your model id"

# Invoke model
try:
    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json"
    )

    # Read and parse response
    response_body = json.loads(response["body"].read())
    response_text = response_body.get("generation", "[No text generated]")
    print(response_text)

except Exception as e:
    print("Error invoking model:", e)
