from langchain_ollama import ChatOllama
from langchain_core.prompts import(ChatPromptTemplate,FewShotChatMessagePromptTemplate)
from langchain_core.output_parsers import StrOutputParser
llm=ChatOllama(model="deepseek-r1:8b", base_url="http://127.0.0.1:11434")
examples=[{"input": "今天天气真好", "output": "积极"},
    {"input": "堵车堵了一小时", "output": "消极"},
    {"input": "考试及格了", "output": "积极"},]
example_prompt=ChatPromptTemplate.from_messages([("human", "{input}"),
    ("ai", "{output}"),])
few_shot=FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "判断下面句子的情绪，只输出“积极”或“消极”。"),
    few_shot,
    ("human", "{input}"),
])
chain = final_prompt | llm | StrOutputParser()
print(chain.invoke({"input":"电脑死机了，文件全丢了"}))