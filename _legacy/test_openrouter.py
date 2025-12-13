import os
from openai import OpenAI
from dotenv import load_dotenv

# Load env to get key
load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")

print(f"Using API Key: {api_key[:5]}...{api_key[-5:] if api_key else 'None'}")

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=api_key,
)

print("Attempting to call openai/gpt-oss-120b:free...")

try:
    print("Fetching available models...")
    # List models
    import requests
    response = requests.get("https://openrouter.ai/api/v1/models")
    if response.status_code == 200:
        models = response.json()['data']
        found = [m['id'] for m in models if 'free' in m['id'].lower()]
        print(f"Found models matching 'free': {found}")
    else:
        print(f"Failed to fetch models: {response.text}")

except Exception as e:
    print(f"\nERROR OCCURRED:")
    print(e)
