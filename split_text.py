from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# 1. 加载 + 分块
loader = TextLoader("test.txt", encoding="utf-8")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""],
)
chunks = splitter.split_documents(docs)
print(f"分块完成：{len(chunks)} 块")

# 2. 向量化 + 存入 Chroma
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://127.0.0.1:11434")
vector_store = Chroma(
    collection_name="day5_kb",
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)
vector_store.add_documents(chunks)
print("已存入向量库")

# 3. 检索：问题变成向量，找最相似的 3 条
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
results = retriever.invoke("机器学习是什么？")

for i, doc in enumerate(results):
    print(f"--- 检索结果 {i+1} ---")
    print(doc.page_content[:80])
    print("来源:", doc.metadata)