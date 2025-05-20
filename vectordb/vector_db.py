import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from typing import List
from langchain.embeddings.base import Embeddings
from embedder import ollama_embedder

load_dotenv(dotenv_path="../.env")

model_name = os.getenv("EMBEDDING_MODEL")
base_api = os.getenv("BASE_API")

documents_directory = "./conspiracy_documents"

documents = []

for filename in os.listdir(documents_directory):
    if filename.endswith(".pdf"):
        try:
            file_path = os.path.join(documents_directory, filename)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        except Exception as e:
            print(f"Skipping {filename} due to error: {e}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    length_function=len
)

chunks = text_splitter.split_documents(documents)
chunks = [chunk for chunk in chunks if len(chunk.page_content.strip()) > 20]

script_dir = os.path.dirname(os.path.abspath(__file__))
persist_directory = os.path.join(script_dir, "..", "chroma")
os.makedirs(persist_directory, exist_ok=True)


# clear the persist directory if it exists
if os.path.exists(persist_directory):
    for filename in os.listdir(persist_directory):
        file_path = os.path.join(persist_directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

embedding_model = ollama_embedder
vectordb = Chroma.from_documents(
    chunks,
    embedding_model,
    persist_directory=persist_directory
)
