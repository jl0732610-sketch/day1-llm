from dotenv import load_dotenv
import os
import requests
import json
import re

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

prompt = """从下面这句话中提取信息，只输出 JSON，不要任何解释或多余文字：
"小明今年25岁，住在上海，职业是程序员"
输出格式：
{"姓名": "", "年龄": 0, "城市": "", "职业": ""}"""

resp = requests.post(
    "https://api.deepseek.com/chat/completions",
    headers={"Authorization": f"Bearer {api_key}"},
    json={"model": "deepseek-chat",
          "messages": [{"role": "user", "content": prompt}]},
    timeout=60,
)
content = resp.json()["choices"][0]["message"]["content"]

# 清理可能出现的 ```json 代码块，再解析
content = re.sub(r"```json\s*|\s*```", "", content).strip()
data = json.loads(content)
print(data)
print("类型是：", type(data))