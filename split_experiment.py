from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("test.txt", encoding="utf-8")
full_text = loader.load()[0].page_content

for chunk_size in [100, 300, 600]:
    for overlap in [0, 50, 100]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""],
        )
        chunks = splitter.split_text(full_text)
        print(f"chunk_size={chunk_size}, overlap={overlap} → {len(chunks)} 块")