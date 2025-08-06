import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Redis
from langchain_core.vectorstores import VectorStoreRetriever

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, "docs", "example_docs.txt")

loader = TextLoader(file_path, encoding="utf-8")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

embedding = OllamaEmbeddings(
    base_url="http://192.168.100.3:11434",
    model="llama3.2:latest"
)

REDIS_URL = "redis://localhost:6379"
INDEX_NAME = "grammar_index"

vectorstore = Redis.from_documents(
    documents=splits,
    embedding=embedding,
    redis_url=REDIS_URL,
    index_name=INDEX_NAME
)

retriever: VectorStoreRetriever = vectorstore.as_retriever()
