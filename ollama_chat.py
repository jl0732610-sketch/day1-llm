import requests

url = "http://127.0.0.1:11434/api/generate"
data = {"model": "deepseek-r1:8b", "prompt": "你好，介绍一下你自己", "stream": False}

resp = requests.post(url, json=data, timeout=120)
print(resp.json()["response"])