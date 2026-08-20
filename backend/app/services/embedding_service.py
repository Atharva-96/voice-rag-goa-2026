import os
from typing import List
from fastembed import TextEmbedding
from app.config import settings

class EmbeddingService:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self._model = None

    @property
    def model(self):
        # Lazy load the model to save startup memory (critical for Render cold starts)
        if self._model is None:
            # Set threads to 1 to minimize memory footprint in constrained environments
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            
            self._model = TextEmbedding(
                model_name=self.model_name,
                threads=1
            )
        return self._model

    def embed_text(self, text: str) -> List[float]:
        """
        Embeds a single string and returns a list of floats.
        """
        # TextEmbedding.embed returns a generator
        embeddings = list(self.model.embed([text]))
        return [float(x) for x in embeddings[0]]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a batch of strings.
        """
        if not texts:
            return []
        embeddings = list(self.model.embed(texts))
        return [[float(x) for x in emb] for emb in embeddings]

# Singleton instance
embedding_service = EmbeddingService()
