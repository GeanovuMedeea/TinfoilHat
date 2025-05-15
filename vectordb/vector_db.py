import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from embedder import ollama_embedder
from langchain_community.vectorstores import Chroma


load_dotenv(dotenv_path="../.env")

documents_directory = "../conspiracy_documents"

documents = []

for filename in os.listdir(documents_directory):
    if filename.endswith(".pdf"):
        file_path = os.path.join(documents_directory, filename)
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len
)

chunks = text_splitter.split_documents(documents)

persist_directory = "../chroma"

# check if the persist directory exists
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

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
