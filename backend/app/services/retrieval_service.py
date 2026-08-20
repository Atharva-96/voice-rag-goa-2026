from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import settings
from app.models import SourceDoc
from app.services.embedding_service import embedding_service

class RetrievalService:
    def __init__(self):
        self._client = None

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            # Connects dynamically based on environment configuration
            if settings.QDRANT_API_URL == ":memory:":
                self._client = QdrantClient(":memory:")
            elif settings.QDRANT_API_KEY:
                self._client = QdrantClient(
                    url=settings.QDRANT_API_URL,
                    api_key=settings.QDRANT_API_KEY
                )
            else:
                self._client = QdrantClient(url=settings.QDRANT_API_URL)
        return self._client

    def search(self, query_text: str, limit: int = 3, min_score: float = 0.3) -> List[SourceDoc]:
        """
        Embeds the query and retrieves the most similar chunks from Qdrant.
        """
        query_vector = embedding_service.embed_text(query_text)
        
        response = self.client.query_points(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query=query_vector,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )

        source_docs = []
        for res in response.points:
            if res.score >= min_score:
                payload = res.payload or {}
                passage_text = payload.get("text", "")
                
                # Retrieve source document details from payload
                source_docs.append(SourceDoc(
                    passage_id=str(res.id),
                    passage=passage_text,
                    score=float(res.score),
                    metadata=payload
                ))
                
        return source_docs

# Singleton instance
retrieval_service = RetrievalService()
