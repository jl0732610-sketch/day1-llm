from dotenv import load_dotenv
import os
import requests

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

resp = requests.post(
    "https://api.deepseek.com/chat/completions",
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "你好，介绍一下你自己"}],
    },
    timeout=60,
)
print(resp.json()["choices"][0]["message"]["content"])