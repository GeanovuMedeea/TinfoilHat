import os

import ollama
from langchain.embeddings.base import Embeddings
from typing import List
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")

model_name = os.getenv("EMBEDDING_MODEL")
base_api = os.getenv("BASE_API")


class OllamaEmbedder(Embeddings):
    def __init__(self, base_url: str = base_api, model: str = model_name):
        self.base_url = base_url
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = ollama.embed(model=self.model, input=texts)
        embeddings = response["embeddings"]
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        response = ollama.embed(model=self.model, input=text)
        embedding = response["embeddings"][0]
        return embedding


ollama_embedder = OllamaEmbedder()


