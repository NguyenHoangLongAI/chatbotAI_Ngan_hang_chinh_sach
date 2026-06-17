from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
from config.settings import settings


class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings"""
        return self.model.encode(texts, normalize_embeddings=True)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode single text to embedding"""
        return self.model.encode([text], normalize_embeddings=True)[0]


# Global embedding model instance
embedding_model = EmbeddingModel()