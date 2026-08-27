from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings,ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
loader=TextLoader("test.txt",encoding="utf-8")
docs=loader.load()
splitter=RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    
)
chunks=splitter.split_documents(docs)
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://127.0.0.1:11434")
vector_store=Chroma(
    collection_name="day6_kb",
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)
vector_store.add_documents(chunks)
retriever=vector_store.as_retriever(search_kwargs={"k":3})
llm = ChatOllama(model="deepseek-r1:8b", base_url="http://127.0.0.1:11434")
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是知识库问答助手，只根据下面的上下文回答。如果上下文没有答案，直接说“文档中没有相关信息”，不要编造。\n上下文：\n{context}"),
    ("human", "问题：{question}"),
])
rag_chain=(
    {
        "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
        "question": RunnablePassthrough(),
    }
    |rag_prompt
    |llm
    |StrOutputParser()
)
for question in ["机器学习是什么？", "今天午饭吃什么？", "RAG 能解决什么问题？"]:
    print(f"\n问：{question}")
    print(f"答：{rag_chain.invoke(question)}")