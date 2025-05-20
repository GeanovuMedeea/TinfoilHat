import os

import ollama
import requests
from langchain.embeddings.base import Embeddings
from typing import List
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")

model_name = os.getenv("EMBEDDING_MODEL")
base_api = os.getenv("BASE_API")


class OllamaEmbedder(Embeddings):
    def __init__(self, base_url: str = base_api, model: str = model_name):
        self.base_api = base_url
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(f"{self.base_api}/embeddings", json={"input": texts})
        data = response.json()

        return [item["embedding"] for item in data["data"]]

    def embed_query(self, text: str) -> List[float]:
        response = requests.post(f"{self.base_api}/embeddings", json={"input": [text]})
        data = response.json()
        print(data)
        return data["data"][0]["embedding"]

ollama_embedder = OllamaEmbedder()


