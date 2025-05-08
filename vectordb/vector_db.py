from langchain.docstore.document import Document
import pandas as pd

from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings

import requests
from typing import List

df = pd.read_csv("conspiracy_dataset.csv")

documents = []
for _, row in df.iterrows():
    prompt = row["Input"]
    answer = row["Response"]
    full_text = f"prompt: {prompt}\nresponse: {answer}"
    documents.append(Document(page_content=full_text))


class LocalServerEmbeddings(Embeddings):
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.model = "text-embedding-nomic-embed-text-v1.5"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(f"{self.base_url}/embeddings", json={"input": texts})
        data = response.json()

        return [item["embedding"] for item in data["data"]]

    def embed_query(self, text: str) -> List[float]:
        response = requests.post(f"{self.base_url}/embeddings", json={"input": [text]})
        data = response.json()
        print(data)
        return data["data"][0]["embedding"]

embedding = LocalServerEmbeddings(base_url="http://localhost:1234/v1")

vectordb = Chroma.from_documents(
    documents=documents,
    embedding=embedding,
    persist_directory="chroma-conspiracy"
)

vectordb.persist()
